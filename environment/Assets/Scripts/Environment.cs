using UnityEngine;
using UnityEngine.UI;
using UnityEditor.SceneManagement;
using System.Collections.Generic;

[RequireComponent(typeof (Task))]
public class Environment : MonoBehaviour {
    [SerializeField]
    Text taskText;

    [SerializeField]
    Text successText;

    [SerializeField]
    Text failureText;

    [SerializeField]
    Text rewardText;

    Task task;

    int elapsed = 0;

    void Start() {
        task = GetComponent<Task>();

        Reward.Set(0.0F);

        if(!PlayerPrefs.HasKey("Task Name")) {
            PlayerPrefs.SetString("Task Name", task.Name());
            Trials.Reset();
        }

        if(PlayerPrefs.GetString("Task Name") != task.Name()) {
            PlayerPrefs.SetString("Task Name", task.Name());
            Trials.Reset();
        }

        PlayerPrefs.SetInt("Elapsed Time", elapsed);

        int successCount = Trials.GetSuccess();
        int failureCount = Trials.GetFailure();

        task.Initialize(successCount, failureCount);

        taskText.text = PlayerPrefs.GetString("Task Name");
        successText.text = "Success: " + successCount;
        failureText.text = "Failure: " + failureCount;
        rewardText.text = "Reward: 0";
    }

    void Update() {
        rewardText.text = "Reward: " + Reward.Get();

        if(task.Success()) {
            task.Reset();

            if(task.Done(Trials.GetSuccess(), Trials.GetFailure())) {
                task.Finish();

                PlayerPrefs.SetInt("Success Count", 0);
                PlayerPrefs.SetInt("Failure Count", 0);
                EditorSceneManager.LoadScene(Scenes.Next());
                return;
            }

            Trials.AddSuccess();

            PlayerPrefs.SetInt("Success Count", Trials.GetSuccess());
            PlayerPrefs.SetInt("Failure Count", Trials.GetFailure());
            EditorSceneManager.LoadScene(EditorSceneManager.GetActiveScene().name);
            return;
        }

        if(task.Failure()) {
            task.Reset();

            Trials.AddFailure();

            PlayerPrefs.SetInt("Success Count", Trials.GetSuccess());
            PlayerPrefs.SetInt("Failure Count", Trials.GetFailure());
            EditorSceneManager.LoadScene(EditorSceneManager.GetActiveScene().name);
            return;
        }

        elapsed += 1;

        PlayerPrefs.SetInt("Elapsed Time", elapsed);
    }
}
