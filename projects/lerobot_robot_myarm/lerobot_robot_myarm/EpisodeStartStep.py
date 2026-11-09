class EpisodeStartStep:
    def __init__(self, tasks):
        self.tasks = list(tasks)

    def __call__(self, robot, episode_index, previous_task=None):
        robot.reset_simulation()

        print(f"\n准备录制 episode {episode_index}")
        for i, task in enumerate(self.tasks, start=1):
            print(f"{i:02d}. {task}")

        while True:
            value = input("选择任务编号，回车沿用上一条任务：").strip()

            if not value and previous_task is not None:
                task = previous_task
                break

            if value.isdigit() and 1 <= int(value) <= len(self.tasks):
                task = self.tasks[int(value) - 1]
                break

            print("请输入有效任务编号")

        print(f"当前任务：{task}")
        input("布置好场景后按回车开始录制：")
        return task