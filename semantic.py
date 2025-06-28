from sentence_transformers import SentenceTransformer, util

# Load the SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, good balance of speed and quality

# Load your transcripts
with open("video_9_groundtruth.txt", "r") as f:
    reference = f.read()

with open("rev_video9.txt", "r") as f:
    hypothesis = f.read()

# Compute embeddings
embedding_ref = model.encode(reference, convert_to_tensor=True)
embedding_hyp = model.encode(hypothesis, convert_to_tensor=True)

# Compute cosine similarity
similarity_score = util.cos_sim(embedding_ref, embedding_hyp).item()
print(f"Semantic similarity: {similarity_score:.4f}")
