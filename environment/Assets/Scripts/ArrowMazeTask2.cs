using UnityEngine;

public class ArrowMazeTask2 : ArroMazeTaskBase {
	int waitedTime = 0;
	bool waited = false;

	public override string Name() { return "Arrow Maze Task 2"; }

	public override void Initialize(int success, int failure) {}

	void Update() {
		float x = agent.transform.position.x;

		if(0.0f <= x && x <= 4.0f) {
			if(!waited && (waitedTime >= 2 * 60)) {
				Reward.Add(2.0F);
				waited = true;
			}
			waitedTime += 1;
		} else {
			waitedTime = 0;
		}
	}
}
