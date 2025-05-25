import numpy as np
import pandas as pd
from model.FAREEncoder import FAREEncoder

class ScreenAlternative:
    # w represents the parameter β
    def __init__(self, fare_encoder=FAREEncoder(), u=0.5, v=0.5, w=0.5, norm_method="softmax"):
        self.fare_encoder = fare_encoder
        self.u = u
        self.v = v
        self.w = w
        self.norm_method = norm_method

    def min_max_norm(self, x):
        return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def normalize(self, x):
        if self.norm_method == "softmax":
            return self.softmax(x)
        elif self.norm_method == "minmax":
            return self.min_max_norm(x)
        else:
            raise ValueError("Unsupported normalization method")

    def compute_internal_consistency(self, df: pd.DataFrame):
        df["Generated Question"] = df["Generated Question"].fillna("")
        df["Reasoning Process"] = df["Reasoning Process"].fillna("")

        questions = df["Generated Question"].tolist()
        reasons = df["Reasoning Process"].tolist()

        q_vecs = np.array([self.fare_encoder.encode_text(q) for q in questions])
        r_vecs = np.array([self.fare_encoder.encode_text(r) for r in reasons])

        center_q = np.mean(q_vecs, axis=0)
        center_r = np.mean(r_vecs, axis=0)

        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

        sim_q = np.array([cosine_sim(vec, center_q) for vec in q_vecs])
        sim_r = np.array([cosine_sim(vec, center_r) for vec in r_vecs])

        c_int = self.u * sim_q + (1 - self.u) * sim_r
        df["Internal Consistency"] = self.normalize(c_int)
        return df

    def compute_external_consistency(self, df: pd.DataFrame):
        df["Ref_Question"] = df["Ref_Question"].fillna("")
        df["answer"] = df["answer"].fillna("")
        df["Generated Question"] = df["Generated Question"].fillna("")
        df["Reasoning Process"] = df["Reasoning Process"].fillna("")

        raw_qs = df["Ref_Question"].tolist()
        raw_as = df["answer"].tolist()
        gen_qs = df["Generated Question"].tolist()
        reasons = df["Reasoning Process"].tolist()

        raw_q_vecs = np.array([self.fare_encoder.encode_text(q) for q in raw_qs])
        raw_a_vecs = np.array([self.fare_encoder.encode_text(a) for a in raw_as])
        gen_q_vecs = np.array([self.fare_encoder.encode_text(q) for q in gen_qs])
        reason_vecs = np.array([self.fare_encoder.encode_text(r) for r in reasons])

        sim_q_a = np.array([
            np.dot(gq, ra) / (np.linalg.norm(gq) * np.linalg.norm(ra) + 1e-8)
            for gq, ra in zip(gen_q_vecs, raw_a_vecs)
        ])

        sim_r_i = np.array([
            np.dot(rv, rq) / (np.linalg.norm(rv) * np.linalg.norm(rq) + 1e-8)
            for rv, rq in zip(reason_vecs, raw_q_vecs)
        ])

        a_exts = self.v * sim_r_i + (1 - self.v) * sim_q_a
        df["External Consistency"] = self.normalize(a_exts)
        return df

    def compute_total_score_and_select_best(self, df: pd.DataFrame):
        df["Total Score"] = self.w * df["External Consistency"] * 10 + (1 - self.w) * df["Internal Consistency"]
        best_row = df.loc[df["Total Score"].idxmax()]
        return best_row, df

    def evaluate_and_save(self, df_path: str):
        df = pd.read_csv(df_path)
        df = self.compute_internal_consistency(df)
        df = self.compute_external_consistency(df)
        best_row, df = self.compute_total_score_and_select_best(df)

        df.to_csv(df_path, index=False, encoding="utf-8-sig")
        print(f"✅ Scores saved to: {df_path}")
        return best_row, df

    def evaluate_by_group(self, df_path: str, output_path: str = None):
        df = pd.read_csv(df_path)

        if "Original Question ID" not in df.columns:
            raise ValueError("Missing 'Original Question ID' column for grouping.")

        best_rows = []

        for raw_id, group in df.groupby("Original Question ID"):
            print(f"🔍 Processing ID: {raw_id}, total {len(group)} candidates")
            group = self.compute_internal_consistency(group)
            group = self.compute_external_consistency(group)
            best_row, _ = self.compute_total_score_and_select_best(group)
            best_rows.append(best_row)

        df_best = pd.DataFrame(best_rows)

        if not output_path:
            output_path = df_path.replace(".csv", "_best.csv")

        df_best.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"🎯 Best results saved to: {output_path}")
        return df_best
