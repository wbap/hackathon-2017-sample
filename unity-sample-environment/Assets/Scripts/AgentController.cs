using UnityEngine;

[RequireComponent(typeof (CharacterController))]
public class AgentController : MonoBehaviour {
    float rotationSpeed = 10.0F;
    float movementSpeed = 1.0F;

    [HideInInspector]
    public bool Paralyzed = false;

    public void Start() {
        rotationSpeed = PlayerPrefs.GetFloat("Rotation Speed");
        movementSpeed = PlayerPrefs.GetFloat("Movement Speed");
    }

    public void PerformAction(string action) {
        CharacterController controller = GetComponent<CharacterController>();

        if(Paralyzed) return;

        switch(action) {
        case "0":
            transform.Rotate(Vector3.up, rotationSpeed);
            break;
        case "1":
            transform.Rotate(Vector3.down, rotationSpeed);
            break;
        case "2":
            Vector3 direction = transform.TransformDirection(Vector3.forward * movementSpeed);
            controller.Move(direction);
            break;
        default:
            break;
        }
    }

}
