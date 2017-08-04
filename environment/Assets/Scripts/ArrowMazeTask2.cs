using UnityEngine;

public class ArrowMazeTask2 : Task {
	int waited = 0;

	public override string Name() { return "Arrow Maze Task 2"; }
	public override string Next() { return "ArrowMazeTask3"; }

	public override void Initialize(int success, int failure) {
	}

	public override bool Success() {
		return rewardCount > 0;
	}

	public override bool Failure() {
		return Reward.Get() < -1.8F;
	}

	public override bool Done(int success, int failure) {
		return (success - failure) > 21;
	}

	void Update() {
		float x = agent.transform.position.z;

		if(0.0f <= x && x <= 2.0f) {
			if(waited >= 2 * 60) {
				rewardCount += 1;
				Reward.Add(2.0F);
			}
			waited += 1;
		} else {
			waited = 0;
		}
		Reward.Add(-0.001F);
	}
}
