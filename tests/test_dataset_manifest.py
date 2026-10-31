import pandas as pd

from mi_bici import dataset
from mi_bici.dataset import compute_content_hash, compute_file_md5, compute_row_hashes


def viajes():
    """Arma una tabla con los mismos tipos que los viajes procesados."""
    return pd.DataFrame(
        {
            "genero": ["M", "F", "NE", "M"],
            "inicio": ["2025-10-01 08:00:00", "2025-10-01 09:30:00", "2025-11-02 18:00:00", "2025-11-03 07:15:00"],
            "origen_id": [10, 20, 10, 30],
            "nacimiento": pd.array([1990, None, 1985, 2001], dtype="Int64"),
            "duracion_min": [12.5, 30.0, 8.25, 15.0],
            "anio": ["2025", "2025", "2025", "2025"],
            "mes": ["10", "10", "11", "11"],
        }
    )


def ingesta(carpeta, compresion, semilla):
    """Escribe estaciones y viajes con otra compresión y otro orden de filas, como otra computadora."""
    estaciones = pd.DataFrame({"estacion_id": [1, 2, 3], "nombre": ["A", "B", "C"], "capacidad": [15, 19, 12]})
    estaciones.sample(frac=1, random_state=semilla).to_parquet(
        carpeta / "estaciones.parquet", index=False, compression=compresion
    )
    viajes().sample(frac=1, random_state=semilla).to_parquet(
        carpeta / "viajes", partition_cols=["anio", "mes"], index=False, compression=compresion
    )


def manifiesto(carpeta, monkeypatch):
    """Genera el manifiesto con las rutas de `carpeta`."""
    monkeypatch.setattr(dataset, "ESTACIONES_FILE", carpeta / "estaciones.parquet")
    monkeypatch.setattr(dataset, "ESTACIONES_METADATA_FILE", carpeta / "estaciones_metadata.json")
    monkeypatch.setattr(dataset, "VIAJES_DIR", carpeta / "viajes")
    monkeypatch.setattr(dataset, "VIAJES_METADATA_FILE", carpeta / "viajes_metadata.json")
    return dataset.generate_manifest(carpeta / "manifest.json")


class TestManifest:
    def test_hash_no_depende_del_orden_de_las_filas(self):
        """Verifica que la misma tabla con las filas en otro orden da el mismo hash."""
        tabla = viajes()
        revuelta = tabla.sample(frac=1, random_state=0)

        assert compute_content_hash(compute_row_hashes(tabla)) == compute_content_hash(
            compute_row_hashes(revuelta)
        )

    def test_hash_cambia_si_cambia_un_valor(self):
        """Verifica que cambiar un solo valor cambia el hash."""
        tabla = viajes()
        otra = tabla.copy()
        otra.loc[0, "duracion_min"] = 12.75

        assert compute_content_hash(compute_row_hashes(tabla)) != compute_content_hash(
            compute_row_hashes(otra)
        )

    def test_manifiesto_igual_aunque_cambien_los_archivos(self, tmp_path, monkeypatch):
        """Verifica que dos ingestas con los mismos datos y otros bytes dan el mismo manifiesto."""
        primera, segunda = tmp_path / "primera", tmp_path / "segunda"
        for carpeta in (primera, segunda):
            carpeta.mkdir()
        ingesta(primera, "snappy", semilla=1)
        ingesta(segunda, "gzip", semilla=2)

        # Los archivos sí son distintos byte por byte
        assert compute_file_md5(primera / "estaciones.parquet") != compute_file_md5(
            segunda / "estaciones.parquet"
        )
        assert manifiesto(primera, monkeypatch) == manifiesto(segunda, monkeypatch)

    def test_coordenadas_con_redondeo_exacto(self, tmp_path, monkeypatch):
        """Verifica que las coordenadas de estaciones se leen igual que float(), en cualquier computadora."""
        textos = ["20.691846603895982", "-103.36805841104507"]
        (tmp_path / "nomenclatura_2026_09.csv").write_text(
            f'"id","name","latitude","longitude","dpcapacity"\n"2","A","{textos[0]}","{textos[1]}","15"\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(dataset, "RAW_DATA_DIR", tmp_path)

        fila = dataset.load_data_stations().iloc[0]

        assert (fila["latitude"], fila["longitude"]) == tuple(float(t) for t in textos)

    def test_manifiesto_sin_md5_ni_tamanos(self, tmp_path, monkeypatch):
        """Verifica que el manifiesto solo guarda lo que depende del contenido, con la partición de cada archivo."""
        ingesta(tmp_path, "snappy", semilla=1)

        datos = manifiesto(tmp_path, monkeypatch)

        for tabla in ("estaciones", "viajes"):
            assert "content_hash" in datos[tabla]
            assert not {"md5", "size_bytes", "total_size_bytes"} & set(datos[tabla])
        assert [f["partition"] for f in datos["viajes"]["files"]] == ["anio=2025/mes=10", "anio=2025/mes=11"]
        assert sum(f["num_rows"] for f in datos["viajes"]["files"]) == datos["viajes"]["total_rows"] == 4
