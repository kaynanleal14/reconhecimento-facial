import json
import os
import sqlite3
import numpy as np


class FaceRecognizer:

    def __init__(self, *args, **kwargs):
        # Limiar oficial recomendado pelo SFace (Cosine Similarity >= 0.363)
        self.threshold = 0.363
        self.db_path = "database/iff_access.db"
        self.db_obj = None
        self._db_empty_warned = False

        # Cache na RAM para busca vetorizada
        self.matrix_cache = None  # Matriz 2D NumPy (N x 128)
        self.users_cache = []  # Lista alinhada de metadados dos usuários

        # Trata parâmetros nomeados (kwargs)
        if "threshold" in kwargs and isinstance(
            kwargs["threshold"], (int, float)
        ):
            self.threshold = float(kwargs["threshold"])
        if "db_path" in kwargs and isinstance(kwargs["db_path"], str):
            self.db_path = kwargs["db_path"]

        # Trata parâmetros posicionais (*args)
        for arg in args:
            if isinstance(arg, (int, float)):
                self.threshold = float(arg)
            elif isinstance(arg, str):
                self.db_path = arg
            elif arg is not None:
                self.db_obj = arg

        # Resolve caminho do banco de dados SQLite
        if not os.path.isabs(self.db_path):
            base_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..")
            )
            candidates = [
                os.path.join(base_dir, "database", "iff_access.db"),
                os.path.join(base_dir, self.db_path),
                os.path.join(base_dir, "iff_access.db"),
                self.db_path,
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    self.db_path = candidate
                    break

        print(
            f"[INFO] FaceRecognizer (limiar={self.threshold}) utilizando o banco: '{self.db_path}'"
        )

        # Monta o cache matricial na RAM na inicialização
        self.reload_cache()

    def _load_users_from_db(self):
        """Carrega usuários e inspeciona colunas em busca dos vetores faciais."""
        if self.db_obj is not None:
            for method in [
                "get_all_users",
                "get_users",
                "get_members",
                "fetch_all",
            ]:
                if hasattr(self.db_obj, method) and callable(
                    getattr(self.db_obj, method)
                ):
                    try:
                        data = getattr(self.db_obj, method)()
                        if data:
                            users = []
                            for u in data:
                                if isinstance(u, dict):
                                    users.append(u)
                                elif hasattr(u, "__dict__"):
                                    users.append(u.__dict__)
                            if users:
                                return users
                    except Exception:
                        pass

        if not os.path.exists(self.db_path):
            return []

        users = []
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            )
            tables = [t[0] for t in cursor.fetchall()]

            if not tables:
                conn.close()
                return []

            for table in tables:
                cursor.execute(f"SELECT * FROM {table}")
                rows = cursor.fetchall()
                col_names = [
                    description[0] for description in cursor.description
                ]

                for row in rows:
                    user_dict = dict(zip(col_names, row))
                    emb_raw = None

                    for key, val in user_dict.items():
                        if val is None:
                            continue
                        if any(
                            kw in key.lower()
                            for kw in [
                                "embed",
                                "vector",
                                "feat",
                                "desc",
                                "vetor",
                                "face",
                                "blob",
                                "data",
                            ]
                        ):
                            emb_raw = val
                            break
                        elif isinstance(val, (bytes, bytearray)) and len(val) >= 128:
                            emb_raw = val
                            break

                    if emb_raw is not None:
                        try:
                            if isinstance(emb_raw, (bytes, bytearray)):
                                arr = np.frombuffer(emb_raw, dtype=np.float32)
                                if len(arr) > 0:
                                    user_dict["embedding"] = arr
                            elif isinstance(emb_raw, str):
                                try:
                                    user_dict["embedding"] = np.array(
                                        json.loads(emb_raw), dtype=np.float32
                                    )
                                except Exception:
                                    user_dict["embedding"] = np.array(
                                        [
                                            float(x)
                                            for x in emb_raw.split(",")
                                            if x.strip()
                                        ],
                                        dtype=np.float32,
                                    )
                            else:
                                user_dict["embedding"] = np.array(
                                    emb_raw, dtype=np.float32
                                )

                            if (
                                "embedding" in user_dict
                                and user_dict["embedding"].size > 0
                            ):
                                users.append(user_dict)
                        except Exception as parse_err:
                            print(
                                f"[WARN] Erro ao converter vetor do registro: {parse_err}"
                            )

            conn.close()
        except Exception as e:
            print(f"[WARN] Erro ao ler banco de dados '{self.db_path}': {e}")

        return users

    def reload_cache(self):
        """Lê o banco, aplica normalização L2 e cria a Matriz 2D em memória RAM."""
        raw_users = self._load_users_from_db()
        if not raw_users:
            self.matrix_cache = None
            self.users_cache = []
            return

        valid_users = []
        normalized_vectors = []

        for u in raw_users:
            emb = (
                u.get("embedding")
                if isinstance(u, dict)
                else getattr(u, "embedding", None)
            )
            if emb is None:
                continue

            vec = np.array(emb, dtype=np.float32).flatten()
            norm = np.linalg.norm(vec)

            # Normalização L2: v / ||v||
            if norm > 0:
                vec = vec / norm
                normalized_vectors.append(vec)
                valid_users.append(u)

        if normalized_vectors:
            self.matrix_cache = np.array(normalized_vectors, dtype=np.float32)
            self.users_cache = valid_users
            kb_size = self.matrix_cache.nbytes / 1024
            print(
                f"[CACHE] {len(valid_users)} rostos indexados na Matriz RAM ({kb_size:.2f} KB)."
            )
        else:
            self.matrix_cache = None
            self.users_cache = []

    

    def recognize(self, embedding, registered_users=None): #Metodo principal de reconhecimento facial, busca vetorizada ultra-rápida usando np.dot para todo o banco de dados em uma única operação.
        #usando metodo de vetorizacao matricial para acelerar a busca de similaridade entre o embedding fornecido e os embeddings armazenados no banco de dados.
        """Busca vetorizada matricial ultra-rápida (1 operação np.dot para todo o banco)."""
        if embedding is None:
            return None, 0.0

        # Se o cache estiver vazio ou a UI passar uma nova lista, recarrega o index
        if self.matrix_cache is None or registered_users is not None:
            self.reload_cache()

        if self.matrix_cache is None or len(self.users_cache) == 0:
            if not self._db_empty_warned:
                print(
                    f"[INFO] Nenhum registro de rosto carregado a partir de '{self.db_path}'."
                )
                self._db_empty_warned = True
            return None, 0.0

        self._db_empty_warned = False

        # Normalização L2 do vetor capturado pela câmera
        query_vec = np.array(embedding, dtype=np.float32).flatten()
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return None, 0.0
        query_vec = query_vec / query_norm

        #  MULTIPLICAÇÃO MATRICIAL EM C: (N x 128) . (128 x 1) -> Vetor de Similaridades (N x 1)
        scores = np.dot(self.matrix_cache, query_vec)

        # Encontra o índice com a maior pontuação de similaridade
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        best_user = self.users_cache[best_idx]
        if isinstance(best_user, dict):
            user_name = (
                best_user.get("nome")
                or best_user.get("name")
                or best_user.get("usuario")
                or best_user.get("id")
                or "Cadastrado"
            )
        else:
            user_name = getattr(
                best_user, "nome", getattr(best_user, "name", "Cadastrado")
            )

        print(
            f"[DEBUG] Maior similaridade: '{user_name}' = {best_score:.4f} (limiar min = {self.threshold})"
        )

        thresh = (
            float(self.threshold)
            if isinstance(self.threshold, (int, float))
            else 0.363
        )

        if best_score >= thresh:
            return best_user, best_score

        return None, best_score