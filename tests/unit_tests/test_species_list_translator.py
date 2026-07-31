from src.species_list_translator import translate_species_list_for_model


def test_translate_birdnet_style_for_perch_v2():
    source = [
        "koala_Phascolarctos cinereus",
        "common cuckoo_Cuculus canorus",
        "Ninox boobook_Morepork",
    ]

    translated = translate_species_list_for_model(source, "perch_v2")

    assert translated == [
        "Phascolarctos cinereus",
        "Cuculus canorus",
        "Ninox boobook",
    ]


def test_translate_scientific_for_perch_8_to_ebird_codes():
    source = [
        "Cuculus canorus",
        "Ninox boobook",
    ]

    translated = translate_species_list_for_model(source, "perch_8")

    assert translated == ["comcuc", "souboo8"]


def test_translate_perch_v2_uses_inat_mapping_fallback_for_synonyms():
    source = [
        "Bubulcus ibis",
    ]

    translated = translate_species_list_for_model(source, "perch_v2")

    assert translated == ["Ardea ibis"]


def test_translate_perch_v2_uses_inat_mapping_fallback_with_compound_entry():
    source = [
        "Bubulcus ibis_Cattle Egret",
    ]

    translated = translate_species_list_for_model(source, "perch_v2")

    assert translated == ["Ardea ibis"]


def test_translate_raises_when_nothing_matches_for_supported_model():
    source = ["not_a_real_species_foo"]

    try:
        translate_species_list_for_model(source, "perch_v2")
    except ValueError as exc:
        assert "No species list entries matched" in str(exc)
    else:
        raise AssertionError("Expected ValueError when no entries translate")
