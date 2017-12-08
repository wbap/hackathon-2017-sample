using UnityEngine;
using System;

public class ArrowMazeTask1 : ArrowMazeTaskBase {
    public GameObject reward;

	int waitedTime = 0;
	bool waited = false;

    public override string AutomationSequence() {
        return String.Join("", new string[] {
            new String('0', 11),
            new String('2', 8),
            new String('3', 130),
            new String('1', 4),
            new String('2', 8)
        });
    }

	public override string Name() { return "Arrow Maze Task 1"; }

	public override void Initialize(int success, int failure) {}

	void Update() {
        float x = agent.transform.position.x;

        if(0.0f <= x && x <= 4.0f) {
            if(!waited && (waitedTime >= 2 * 60)) {
                Reward.Add(2.0F);

                GameObject rewardObj = (GameObject)GameObject.Instantiate(
                    reward, new Vector3(10.56F, 0.5F, 8.05F), Quaternion.identity
                );

                waited = true;
            }
            waitedTime += 1;
        } else {
            waitedTime = 0;
        }
	}
}
