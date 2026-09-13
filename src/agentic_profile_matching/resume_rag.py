import re
from pathlib import Path
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer

from agentic_profile_matching import config
from agentic_profile_matching.observability import get_logger
from agentic_profile_matching.fs_client import list_files, read_file
from agentic_profile_matching.stores import BaseVectorStore, ChromaVectorStore

logger = get_logger("agentic_profile_matching.resume_rag")

SECTION_HEADERS = [
    "EXPERIENCE",
    "WORK EXPERIENCE",
    "EDUCATION",
    "SKILLS",
    "PROJECTS",
    "CERTIFICATIONS",
    "SUMMARY",
    "PROFESSIONAL EXPERIENCE",
    "ACADEMIC BACKGROUND",
    "TECHNICAL SKILLS",
    "AWARDS",
    "WORK HISTORY",
    "OBJECTIVE",
]


class MetadataExtractor:
    SKILLS = [
        "Python",
        "Java",
        "Spring Boot",
        "AWS",
        "Docker",
        "Kubernetes",
        "SQL",
        "Kafka",
        "React",
        "JavaScript",
        "TypeScript",
        "HTML",
        "CSS",
        "C++",
        "Go",
        "Golang",
        "Rust",
        "Ruby",
        "Rails",
        "PHP",
        "Laravel",
        "Swift",
        "Kotlin",
        "Objective-C",
        "Android",
        "iOS",
        "Flutter",
        "React Native",
        "Node.js",
        "Express",
        "Django",
        "Flask",
        "FastAPI",
        "PostgreSQL",
        "MySQL",
        "MongoDB",
        "Redis",
        "Cassandra",
        "Elasticsearch",
        "Spark",
        "Hadoop",
        "Pandas",
        "NumPy",
        "Scikit-Learn",
        "TensorFlow",
        "PyTorch",
        "Git",
        "CI/CD",
        "Jenkins",
        "Terraform",
        "Ansible",
        "Linux",
        "GCP",
        "Azure",
        "HTML5",
        "CSS3",
        "GraphQL",
        "REST API",
        "Microservices",
        "Machine Learning",
        "Deep Learning",
        "NLP",
        "Generative AI",
        "LLM",
        "LangChain",
        "RAG",
        "Data Science",
        "Cloud",
        "Cloud Computing",
        "DevOps",
        "Backend",
        "Frontend",
        "Full Stack",
        "Distributed Systems",
    ]

    def extract_name(self, filename: str, text: str) -> str:
        # Heuristic 1: Clean filename to extract name if format matches common patterns
        base = Path(filename).stem
        for prefix in ["resume_", "resume-", "cv_", "cv-", "summary_", "summary-"]:
            if base.lower().startswith(prefix):
                base = base[len(prefix) :]
        for suffix in ["_resume", "-resume", "_cv", "-cv", "_profile", "-profile"]:
            if base.lower().endswith(suffix):
                base = base[: -len(suffix)]
        if "_" in base or "-" in base or " " in base:
            name = base.replace("_", " ").replace("-", " ").title()
            name = " ".join(part for part in name.split() if part.isalpha())
            if name:
                return name

        # Heuristic 2: Extract from the first few lines of text
        for line in text.splitlines():
            line_stripped = line.strip()
            if not line_stripped:
                continue
            if line_stripped.upper() in SECTION_HEADERS:
                continue
            words = line_stripped.split()
            if 1 <= len(words) <= 4 and all(w.replace(".", "").isalpha() for w in words):
                return line_stripped.title()
            break
        return "Unknown"

    def extract_skills(self, text: str) -> List[str]:
        text_lower = text.lower()
        found = []
        for skill in self.SKILLS:
            pattern = rf"\b{re.escape(skill.lower())}\b"
            if not skill.isalnum():
                if skill.lower() in text_lower:
                    found.append(skill)
            else:
                if re.search(pattern, text_lower):
                    found.append(skill)

        # Open-ended extraction: parse explicit SKILLS / TECHNICAL SKILLS section
        # Scope case-insensitivity strictly to the section title, keeping header terminators uppercase-only
        skills_match = re.search(
            r"(?i:TECHNICAL\s+SKILLS|SKILLS)\s*[:\n](.*?)(?:\n\s*[A-Z][A-Z\s]{3,}[:\n]|\Z)",
            text,
            re.DOTALL,
        )
        if skills_match:
            section_content = skills_match.group(1).strip()
            raw_tokens = re.split(r"[,•\n|;]", section_content)
            for tok in raw_tokens:
                clean_tok = tok.strip().strip(".-* ")
                # Strip HTML tags/markup for security
                clean_tok = re.sub(r"<[^>]+>", "", clean_tok).strip()
                if clean_tok and 1 < len(clean_tok) <= 40 and not any(h in clean_tok.upper() for h in SECTION_HEADERS):
                    # Canonicalize title case if all lowercase or mixed
                    found.append(clean_tok)

        return sorted(list(set(found)))

    def extract_experience(self, text: str) -> int:
        patterns = [
            r"(\d+)\+?\s*(?:years?|yrs?)(?:\s*of)?\s*(?:experience|exp)",
            r"(?:experience|exp)\s*:\s*(\d+)\+?\s*(?:years?|yrs?)",
            r"(\d+)\+?\s*(?:years?|yrs?)\b",
        ]
        years = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.I)
            for m in matches:
                try:
                    years.append(int(m))
                except ValueError:
                    pass
        # Validate years of experience to cap it at 50 to filter out phone numbers/postcodes/years (e.g. 2024)
        valid_years = [y for y in years if 0 <= y <= 50]
        return max(valid_years, default=0)

    def extract_education(self, text: str) -> str:
        edu_keywords = [
            "B.S.",
            "B.S",
            "B.Tech",
            "M.S.",
            "M.S",
            "M.Tech",
            "Ph.D.",
            "PhD",
            "Bachelor",
            "Master",
            "B.A.",
            "B.A",
            "M.A.",
            "M.A",
        ]
        education_entries = []

        for line in text.splitlines():
            line_stripped = line.strip()
            for kw in edu_keywords:
                if re.search(rf"\b{re.escape(kw)}\b", line_stripped, re.I):
                    education_entries.append(line_stripped)
                    break

        if education_entries:
            return "; ".join(education_entries[:3])
        return "Not Specified"

    def extract(self, filename: str, text: str) -> Dict:
        return {
            "candidate_name": self.extract_name(filename, text),
            "skills": self.extract_skills(text),
            "experience_years": self.extract_experience(text),
            "education": self.extract_education(text),
        }


class ResumeChunker:
    def chunk(self, text: str) -> List[Dict[str, str]]:
        chunks = []
        current_section = "GENERAL"
        buf = []

        for line in text.splitlines():
            line_clean = line.strip().strip("#").strip().strip(":").strip()
            if line_clean.upper() in SECTION_HEADERS:
                if buf:
                    content = "\n".join(buf).strip()
                    if content:
                        chunks.append({"section": current_section, "content": content})
                current_section = line_clean.upper()
                buf = []
            else:
                buf.append(line)

        if buf:
            content = "\n".join(buf).strip()
            if content:
                chunks.append({"section": current_section, "content": content})

        return chunks


class ResumeRAGPipeline:
    def __init__(
        self,
        store: Optional[BaseVectorStore] = None,
        model_name: Optional[str] = None,
        collection_name: str = "resumes",
    ):
        self.store = store or ChromaVectorStore(collection_name=collection_name)
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.embedder = SentenceTransformer(self.model_name)

    def ingest_directory(self, resume_dir: str):
        extractor = MetadataExtractor()
        chunker = ResumeChunker()
        files = list_files(resume_dir)

        for f in files:
            data = read_file(f["path"])
            if not data.get("success"):
                continue
            text = data["content"]

            meta = extractor.extract(f["name"], text)
            chunks = chunker.chunk(text)

            # Print for ingestion tracking
            print(
                f"Ingesting {f['name']} - Name: {meta['candidate_name']}, Exp: {meta['experience_years']} yrs, Skills: {len(meta['skills'])}"
            )

            for idx, ch in enumerate(chunks):
                emb = self.embedder.encode(ch["content"]).tolist()
                section_clean = ch["section"].lower().replace(" ", "_")
                chunk_id = f"{f['name']}_{section_clean}_{idx}"

                chunk_meta = {
                    "candidate_name": meta["candidate_name"],
                    "skills": ", ".join(meta["skills"]),
                    "experience_years": int(meta["experience_years"]),
                    "education": meta["education"],
                    "resume_path": f["path"],
                    "filename": f["name"],
                    "section": ch["section"],
                }

                self.store.upsert(
                    ids=[chunk_id],
                    documents=[ch["content"]],
                    embeddings=[emb],
                    metadatas=[chunk_meta],
                )


if __name__ == "__main__":
    import sys

    pipeline = ResumeRAGPipeline()
    directory = sys.argv[1] if len(sys.argv) > 1 else config.RESUMES_DIR
    logger.info(f"Ingesting resumes from directory: {directory}")
    pipeline.ingest_directory(directory)
    logger.info("Ingestion complete.")
