using UnityEngine;
using UnityEditor.SceneManagement;

public class Startup : MonoBehaviour {
    [SerializeField]
    float RotationSpeed;

    [SerializeField]
    float MovementSpeed;

    void Start () {
        PlayerPrefs.DeleteAll();
        PlayerPrefs.SetFloat("Rotation Speed", RotationSpeed);
        PlayerPrefs.SetFloat("Movement Speed", MovementSpeed);
        EditorSceneManager.LoadScene("ArrowMazeTask1");
    }
}
