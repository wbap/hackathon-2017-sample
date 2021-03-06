﻿using UnityEngine;
using System;

public class OneDimTask1 : OneDimTaskBase {
    public override string AutomationSequence() {
        return new String('2', 11);
    }

    public override string Name() { return "One Dimensional Task 1"; }

    public override void Initialize(int success, int failure) {
        float z = (float)(22 - (success - failure));
        agent.transform.position = new Vector3(0.0F, 1.12F, z);
    }

    public override bool Success() {
        return rewardCount > 0;
    }
}
