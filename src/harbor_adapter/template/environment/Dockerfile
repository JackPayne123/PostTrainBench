# Per-task Dockerfile (thin layer over jackpayne123/ptb-base).
#
# All runtime deps (CUDA, vLLM 0.11.0 with transformers<5.0 pinned, claude-code,
# inspect_evals, ML stack) live in the base image. This Dockerfile only adds
# the per-task files. Per-task image build is ~5s after the base is cached.
#
# To rebuild the base image, see dockerfiles/Dockerfile.base in the repo root.
FROM jackpayne123/ptb-base:1

# Workspace dir already exists in the base image, but we WORKDIR it here for clarity.
WORKDIR /home/agent/workspace

# Copy per-task files (populated by adapter; .dockerignore excludes Dockerfile)
COPY . /home/agent/workspace/

# Permissions
RUN chmod -R a+rw /home/agent/workspace/ && \
    chmod +x /home/agent/workspace/timer.sh
