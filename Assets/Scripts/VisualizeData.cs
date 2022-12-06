using System.Collections;
using System.Collections.Generic;
using System.IO;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class VisualizeData : MonoBehaviour
{
    void Update()
    {
        GetComponent<TextMeshProUGUI>().text = string.Format(
            "Graph Data: {0}\nMesh Data: {1}",
            GameSettingsSingleton.Instance.graphJsonString,
            GameSettingsSingleton.Instance.meshJsonString
        );

        if (Input.GetKeyDown(KeyCode.Return))
        {
            _SpawnMesh();
        }
    }

    private void _SpawnMesh()
    {
        GameObject go = Resources.Load("mesh") as GameObject;
        Instantiate(go, Vector3.zero, Quaternion.identity);
    }

    private void _CreateMeshFromString(string objString)
    {
        Mesh mesh = new Mesh();
        GameObject go = new GameObject("Mesh", typeof(MeshFilter), typeof(MeshRenderer));
        MeshFilter meshFilter = go.GetComponent<MeshFilter>();
        //meshFilter.mesh = new OBJLoader()
    }
}
