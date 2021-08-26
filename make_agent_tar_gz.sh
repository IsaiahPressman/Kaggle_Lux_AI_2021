archive_name="${1:-"submission.tar.gz"}"
tar --exclude="lux_ai/rl_agent_2" --exclude="*__pycache__*" -czvf "$archive_name" main.py __init__.py lux_ai
echo "Agent saved to $archive_name"