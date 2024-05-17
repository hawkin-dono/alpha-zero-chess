import concurrent.futures
import time 
import threading

num_parallel_games = 100
nums_episodes = 10000

def play_game(game_id):
    return game_id
    
res = [] 
start = time.time()
for i in range(nums_episodes):
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel_games)
    futures = []
    for i in range(num_parallel_games):
        future = pool.submit(play_game, i)
        futures.append(future)
    for future in futures:
        res.append(future.result())
    pool.shutdown(wait=True, cancel_futures=True)

print(f'Pool Time taken: {time.time() - start} seconds')

res = []
start = time.time()
for i in range(nums_episodes):
    threads = []
    for i in range(num_parallel_games):
        t = threading.Thread(target=play_game, args=(i,))
        t.start()
        threads.append(t)
    for thread in threads:
        thread.join()

print(f'Thread Time taken: {time.time() - start} seconds')

res = []
start = time.time()
for i in range(nums_episodes * num_parallel_games):
    res.append(i)
    
print(f'Normal Time taken: {time.time() - start} seconds')
    


        