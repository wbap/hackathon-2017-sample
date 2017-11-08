using UnityEngine;

public class OneDimTask1 : OneDimTaskBase {
    public override string Name() { return "One Dimensional Task 1"; }

    public override void Initialize(int success, int failure) {
        float z = (float)(22 - (success - failure));
        agent.transform.position = new Vector3(0.0F, 1.12F, z);
    }
}
