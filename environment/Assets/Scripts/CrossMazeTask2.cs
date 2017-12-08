using UnityEngine;
using System;

public class CrossMazeTask2 : CrossMazeTaskBase {
	string automation;

	public override string AutomationSequence() { return automation; }

	public override string Name() { return "Cross Maze Task 2"; }

	public override void Initialize(int success, int failure) {
		// 仕様「S地点は中心で、向きはランダム」を実現する

		int phase = (int)(UnityEngine.Random.value * 4);
		float rx = 0.0f;
		float ry = 0.0f;
		float rz = 0.0f;
		Quaternion rotation = Quaternion.identity;

		switch(phase) {
		case 0:
			// 北向きでスタート {0, 0, 0}
			automation = String.Join("", new string[] {
				new String('1', 9),
				new String('2', 6),
				new String('0', 18),
				new String('2', 6),

				new String('1', 9),
				new String('2', 6),
				new String('0', 18),
				new String('2', 6),

				new String('1', 9),
				new String('2', 6)
			});
			ry = 0.0f;
			break;
		case 1:
			// 西向きでスタート {n0, 0, 0}
			automation = String.Join("", new string[] {
				new String('2', 6),
				new String('0', 18),
				new String('2', 6),

				new String('1', 9),
				new String('2', 6),
				new String('0', 18),
				new String('2', 6),

				new String('1', 9),
				new String('2', 6)
			});
			ry = -90.0f;
			break;
		case 2:
			// 東向きでスタート {0, 0, 0}
			automation = String.Join("", new string[] {
				new String('0', 18),
				new String('2', 6),
				new String('0', 18),
				new String('2', 6),

				new String('1', 9),
				new String('2', 6),
				new String('0', 18),
				new String('2', 6),

				new String('1', 9),
				new String('2', 6)
			});
			ry = 90.0f;
			break;
		case 3:
			// 向きでスタート {0, 0, 0}
			automation = String.Join("", new string[] {
				new String('0', 9),
				new String('2', 6),
				new String('0', 18),
				new String('2', 6),

				new String('1', 9),
				new String('2', 6),
				new String('0', 18),
				new String('2', 6),

				new String('1', 9),
				new String('2', 6)
			});
			ry = 180.0f;
			break;
		default:
			break;
		}

		rotation.eulerAngles = new Vector3 (rx, ry, rz);
		agent.transform.rotation = rotation;
	}

	public override bool Success() {
		return rewardCount > 2;
	}
}
