public abstract class OneDimTaskBase : Task {
    public override bool Success() {
        return rewardCount > 1;
    }

    public override bool Failure() {
        return Reward.Get() < -1.8F;
    }

    public override bool Done(int success, int failure) {
        return (success - failure) > 21;
    }
}
