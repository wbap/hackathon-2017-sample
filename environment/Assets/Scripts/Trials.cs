using UnityEngine;

public class Trials {
    public static void Reset() {
        Debug.Log("Reset trials");
    }

    public static void AddSuccess() {
        string trials = PlayerPrefs.GetString("Trials");

        trials += "S";

        while(trials.Length > 100) {
            trials = trials.Substring(1);
        }

        PlayerPrefs.SetString("Trials", trials);
    }

    public static void AddFailure() {
        string trials = PlayerPrefs.GetString("Trials");

        trials += "F";

        while(trials.Length > 100) {
            trials = trials.Substring(1);
        }

        PlayerPrefs.SetString("Trials", trials);
    }

    public static int GetSuccess() {
        int count = 0;
        string trials = PlayerPrefs.GetString("Trials");

        foreach(char trial in trials) {
            if(trial == 'S') {
                count += 1;
            }
        }

        return count;
    }

    public static int GetFailure() {
        int count = 0;
        string trials = PlayerPrefs.GetString("Trials");

        foreach(char trial in trials) {
            if(trial == 'F') {
                count += 1;
            }
        }

        return count;
    }
}
