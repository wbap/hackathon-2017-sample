using UnityEngine;

public class Automator : MonoBehaviour {
    public static void Setup(string sequence) {
        PlayerPrefs.SetString("Automation Sequence", sequence);
    }

    public static string Step() {
        string sequence = PlayerPrefs.GetString("Automation Sequence");

        if(sequence.Length > 0) {
            string head = sequence.Substring(0, 1);
            string tail = sequence.Substring(1);
            PlayerPrefs.SetString("Automation Sequence", tail);
            return head;
        }

        return "";
    }

    public static bool Enabled() {
        return PlayerPrefs.GetInt("Autorun") != 0;
    }
}
