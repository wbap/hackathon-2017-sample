using UnityEngine;
using System;

public class OneDimTask3 : OneDimTaskBase {
    public GameObject reward;

    bool rewardShown = false;
    int waited = 0;

    Range range = Range.Green;

    public override string AutomationSequence() {
        string[] sequences = new string[] {
            new String('2', 7),
            new String('3', 120),
            new String('2', 4)
        };

        return String.Join("", sequences);
    }

    public override string Name() { return "One Dimensional Task 3"; }

    void FixedUpdate() {
        float z = agent.transform.position.z;

        if(range.start <= z && z <= range.end) {
            if(!rewardShown && waited >= 2 * 60) {
                rewardCount += 1;
                Reward.Add(2.0F);

                GameObject rewardObj = (GameObject)GameObject.Instantiate(
                    reward, new Vector3(0.0F, 0.5F, 23.0F), Quaternion.identity
                );

                rewardObj.transform.parent = transform;
                rewardShown = true;
            }

            waited += 1;
        } else {
            waited = 0;
        }
    }
}
