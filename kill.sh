ps aux | grep gpt | awk '{print $2}' | xargs kill -9
