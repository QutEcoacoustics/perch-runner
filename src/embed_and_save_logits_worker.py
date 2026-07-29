import threading
import itertools
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent import futures
from perch_hoplite.agile import embed as agile_embed
from perch_hoplite.taxonomy import namespace_db


def _species_key_candidates(model_choice):
    if model_choice == 'perch_8':
        return ('label', 'labels')
    if model_choice == 'perch_v2':
        return ('labels', 'label')
    return ('label', 'labels')


def _extract_species_codes_from_class_list(class_list, model_choice):
    """Return logits-aligned species identifiers as a list of strings."""
    if isinstance(class_list, dict):
        for key in _species_key_candidates(model_choice):
            labels_obj = class_list.get(key)
            if labels_obj is not None and hasattr(labels_obj, 'classes'):
                return [str(c) for c in labels_obj.classes]

        flat_classes = class_list.get('classes', [])
        return [str(c) for c in flat_classes]

    if hasattr(class_list, 'classes'):
        return [str(c) for c in class_list.classes]

    return []


def _load_ebird2021_code_to_scientific_name():
    """Return eBird species code -> Clements scientific name mapping."""
    db = namespace_db.load_db()
    clements_to_species = db.mappings.get('ebird2021_clements_to_species')
    if clements_to_species is None:
        return {}

    # Reverse clements->species mapping into species->clements lookup.
    return {
        species_code: scientific_name
        for scientific_name, species_code in clements_to_species.mapped_pairs.items()
    }


def resolve_species_class_names(
    class_list,
    model_choice,
    ebird_code_to_name=None,
):
    """Resolve logits-aligned species names for either perch_8 or perch_v2.

    For perch_8, species IDs are typically eBird codes. If a mapping is
    available, codes are converted to display names.
    """
    species_codes = _extract_species_codes_from_class_list(class_list, model_choice)

    if model_choice != 'perch_8':
        return species_codes

    if ebird_code_to_name is None:
        ebird_code_to_name = _load_ebird2021_code_to_scientific_name()
    if not ebird_code_to_name:
        return species_codes

    return [ebird_code_to_name.get(code, code) for code in species_codes]


def resolve_species_class_names_for_model_choice(model_choice, ebird_code_to_name=None):
    from perch_hoplite.zoo import model_configs

    preset_model = model_configs.load_model_by_name(model_choice)
    class_list = getattr(preset_model, 'class_list', None)
    return resolve_species_class_names(
        class_list=class_list,
        model_choice=model_choice,
        ebird_code_to_name=ebird_code_to_name,
    )


def _select_top_indices_above_threshold(window_logits, threshold, top_n):
    """Return class indices with scores >= threshold, capped to top_n by score."""
    candidate_indices = np.where(window_logits >= threshold)[0]
    if len(candidate_indices) == 0:
        return candidate_indices

    candidate_scores = window_logits[candidate_indices]
    order = np.argsort(candidate_scores)[::-1]
    if top_n is not None:
        order = order[:top_n]
    return candidate_indices[order]


def process_source_id_with_logits(state, source_id, window_size_s):
    """
    Modified module-level worker to return embeddings AND logits.
    Based on the original process_source_id function in perch_hoplite.agile.agile_embed, 
    with extra stuff added to return logits for each embedding window.
    
    """
    worker = state['worker']
    glob = worker.audio_globs[source_id.dataset_name]
    target_sample_rate = worker.get_sample_rate_hz(source_id)
    audio_array = worker.load_audio(source_id)

    if audio_array is None:
        return
    if (
        audio_array.shape[0]
        < glob.min_audio_len_s * worker.embedding_model.sample_rate
    ):
        return

    outputs = worker.embedding_model.embed(audio_array)
    
    # Force default embedding extraction
    embeddings = outputs.embeddings
    if embeddings is None:
        return
        
    logits_array = None
    if worker.classifier_output_path:
        # Extract species logits only when classify output is enabled.
        logits_dict = outputs.logits or {}
        logits_array = logits_dict.get(worker.logits_key)
        if logits_array is None and logits_dict:
            logits_array = next(iter(logits_dict.values()))

    sources = []
    offsets = []
    embs = []
    logts = []

    hop_size_s = worker.compute_hop_size_s(source_id, target_sample_rate)
    for t, embedding in enumerate(embeddings):
        offset_s = source_id.offset_s + t * hop_size_s
        offsets_list = [offset_s, offset_s + window_size_s]
        for channel_idx, channel_embedding in enumerate(embedding):
            sources.append(source_id)
            offsets.append(offsets_list)
            embs.append(channel_embedding)
            if logits_array is not None:
                # Handle either [time, channels, classes] or [time, classes] logits.
                if logits_array.ndim >= 3:
                    logts.append(logits_array[t, channel_idx])
                else:
                    logts.append(logits_array[t])

    return sources, offsets, embs, logts


class LogitSavingWorker(agile_embed.EmbedWorker):
    def __init__(self, model_choice, logit_threshold=0.0, classifier_output_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parquet_records = []
        self.lock = threading.Lock()
        self.classifier_output_path = classifier_output_path
        #self.class_list = list(getattr(self.embedding_model, 'class_list', []) or [])



        # save logits above this threshold
        self.logit_threshold = logit_threshold
        self.max_classes_per_segment = 10
        self.logits_key = 'label'

        self.class_names = resolve_species_class_names_for_model_choice(
            model_choice=model_choice,
        )
        # self.class_list = preset_model.class_list


    def embed_dataset(
        self,
        batch_size=32,
        handle_duplicates='error',
        target_dataset_name=None,
        new_recordings=None,
    ):
        """Overridden to use the custom process_source_id_with_logits function."""
        if new_recordings is None:
            new_recordings = set()

        state = {}
        state['db'] = self.db
        state['worker'] = self
        state['new_recordings'] = new_recordings
        
        with futures.ThreadPoolExecutor(
            max_workers=self.audio_worker_threads,
            initializer=agile_embed.worker_initializer,
            initargs=(state,),
        ) as executor:
            source_iterator = self.audio_sources.iterate_all_sources(
                target_dataset_name
            )
            for source_ids_batch in agile_embed.batched(source_iterator, batch_size):
                got = executor.map(
                    process_source_id_with_logits,
                    itertools.repeat(state),
                    source_ids_batch,
                    itertools.repeat(self.window_size_s),
                )
                
                for result in got:
                    if result is None:
                        continue
                        
                    # Unpack the modified 4-tuple return
                    sources, offsets, embs, logts = result
                    recording_ids = []
                    
                    for s in sources:
                        deployment_id = self._get_or_insert_deployment_id(
                            s.deployment_name_from_file_id(), s.dataset_name
                        )
                        recording_id, _ = self._get_or_insert_recording_id(
                            s.file_id, deployment_id, s.dataset_name
                        )
                        recording_ids.append(recording_id)
                        
                    if all(r in new_recordings for r in recording_ids):
                        dupe_strategy = 'allow'
                    else:
                        dupe_strategy = handle_duplicates
                        
                    windows_batch = [
                        {
                            'recording_id': recording_ids[i],
                            'offsets': o,
                        }
                        for i, o in enumerate(offsets)
                    ]
                    embeddings_batch = np.array(embs)
                    
                    # Insert embeddings into database (Original functionality)
                    self.db.insert_windows_batch(
                        windows_batch,
                        embeddings_batch,
                        handle_duplicates=dupe_strategy,
                    )
                    
                    # todo: we might get some confusing behaviour if there is an existing database and we are adding new embeddings to it. 
                    # we will only get predictions for the new sources.  
                    if logts and self.classifier_output_path:
                        with self.lock:
                            for i, s in enumerate(sources):
                                window_id = f"{s.file_id}_{offsets[i][0]}"
                                window_logits = logts[i]
                                above_thresh_indices = _select_top_indices_above_threshold(
                                    window_logits=window_logits,
                                    threshold=self.logit_threshold,
                                    top_n=self.max_classes_per_segment,
                                )
                                for idx in above_thresh_indices:
                                    score = float(window_logits[idx])

                                    # will crash if the classlist length doesn't match the logits length, I think that's ok
                                    if idx < len(self.class_names):
                                        species_name = self.class_names[idx]
                                    else:
                                        species_name = f"class_{idx}"
                                    self.parquet_records.append({
                                        "window_id": window_id,
                                        "recording_id": str(recording_ids[i]),
                                        "offset_s": offsets[i][0],
                                        "species": species_name,
                                        "score": score,
                                    })
                            
        self.db.commit()

        # Keep classify staging in sync with DB lifecycle: write immediately
        # after a successful commit.
        if self.classifier_output_path:
            self.export_parquet(self.classifier_output_path)

    def export_parquet(self, filepath):
        """Export the captured records to Parquet."""
        if not self.parquet_records:
            return

        table = pa.Table.from_pylist(self.parquet_records)
        pq.write_table(table, filepath)