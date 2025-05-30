#!/bin/bash

tmux new-session -d -s mysession

tmux split-window -h -t mysession 'cd backend && source .venv/bin/activate && uvicorn main:app --reload --log-level debug'
tmux split-window -v -t mysession 'cd backend && source .venv/bin/activate && celery -A main.celery_app worker --loglevel=info --concurrency=3'
tmux split-window -v -t mysession 'cd frontend && npm run dev'

tmux select-layout tiled

tmux attach-session -t mysession