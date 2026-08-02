from src.output_paths import render_output_relative_path


def test_render_output_relative_path_uses_extensionless_filestem():
    rel = render_output_relative_path(
        template="{parents}/{filestem}{ext}",
        audio_file="some/folder/myfile.wav",
        analysis="embeddings",
        ext="csv",
        template_type="embeddings",
    )

    assert rel.as_posix() == "some/folder/myfile.csv"


def test_render_output_relative_path_keeps_internal_dots_in_filestem():
    rel = render_output_relative_path(
        template="{filestem}{ext}",
        audio_file="my.file.name.wav",
        analysis="embeddings",
        ext="parquet",
        template_type="embeddings",
    )

    assert rel.as_posix() == "my.file.name.parquet"
