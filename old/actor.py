
def actor(q , stats, queue, e=0.01):
    localQ = R2D2Net(4)
    localQ.load_state_dict(q.state_dict())
    localFrameNumber = 0
    
    while True:
        env = gym.make(ENV_NAME)
        #env = gym.wrappers.Monitor(env, 'videos',video_callable=lambda episode_id: True, force = True)
        obs = env.reset()
        obs = preprocess(obs).unsqueeze(0)
        done = False
        score = 0

        history = localQ.getNewHistory()
        buffers = [((history[0].clone().detach(), history[1].clone().detach() ), [])]
        index = 0
        while not done and index < MAX_GAME_LENGTH:            
            action, history = localQ(obs, history)
            action = action.squeeze().argmax().item()

            if random.random() < e:
                action = env.action_space.sample()

            oldObs = obs
            for _ in range(NUM_ACTION_REPEATS):
                obs,reward,done,inf = env.step(action)
                if done:
                    break
            obs = preprocess(obs).unsqueeze(0)

            score+=reward #update total undiscounted reward for an episode

            localFrameNumber += 1
            with stats.frameNumber.get_lock():
                stats.frameNumber.value += 1

            if localFrameNumber % COPY_NETWORK_FREQUENCY:
                localQ.load_state_dict(q.state_dict())

            index += 1
            for x in range(len(buffers)):
                buffers[x][1].append((oldObs.squeeze(0).numpy(), obs.squeeze(0).numpy(), action, reward, done))

            for x in range(len(buffers)):
                if len(buffers[x][1]) >= EPISODE_LENGTH:
                    queue.put(buffers.pop(x))
                    break #cant be more than one to remove per frame
                    #push to queue

            if index % EPISODE_OVERLAP == 0:
                buffers += [((history[0].clone().detach(), history[1].clone().detach() ), [])]

        with stats.gameNumber.get_lock():
            stats.gameNumber.value += 1
            if stats.gameNumber.value % LOGGING_BATCH_SIZE == 0:
                visdomUpdate(stats)

        with stats.totalScore.get_lock():
            stats.totalScore.value += score

        if score > stats.bestScore.value:
            stats.bestScore.value = score