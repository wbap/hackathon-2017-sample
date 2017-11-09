using UnityEngine;

public class OneDimTask2 : OneDimTaskBase {
    public GameObject reward;

    bool rewardShown = false;

    public override string Name() { return "One Dimensional Task 2"; }

    void Update() {
        float z = agent.transform.position.z;

        if(11.5F <= z && z <= 15.5F) {
            if(!rewardShown) {
                rewardCount += 1;
                Reward.Add(2.0F);

                GameObject rewardObj = (GameObject)GameObject.Instantiate(
                    reward, new Vector3(0.0F, 0.5F, 23.0F), Quaternion.identity
                );

                rewardObj.transform.parent = transform;
                rewardShown = true;
            }
        }
    }
}
