using Photon.Pun;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using TMPro;
using UnityEngine;

public class SendReceive : MonoBehaviourPun, IPunObservable
{
    private PhotonView _photonView;

    private void OnEnable()
    {
        _photonView = GetComponent<PhotonView>();
        if (!_photonView.IsMine) return;

        // set name as 'server' or 'client' from game settings
        _photonView.RPC("PunRPC_SetNickName", RpcTarget.AllBuffered, PhotonNetwork.NickName);
    }
    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space)) // TODO send through UI buttons
        {
            if (PhotonNetwork.NickName == "client")
            {
                //TODO update graph.json from tracked objects
                string path = Application.dataPath + "/Scripts/graph.json";
                string jsonString = File.ReadAllText(path);
                _photonView.RPC("PunRPC_sendGraph", RpcTarget.AllBuffered, jsonString);
            }
            else if (PhotonNetwork.NickName == "server")
            {
                //TODO generate mesh from rhino
                string path = Application.dataPath + "/Resources/mesh.obj";
                string objString = File.ReadAllText(path);
                _photonView.RPC("PunRPC_sendMesh", RpcTarget.AllBuffered, objString);
            }

        }
    }

    [PunRPC]
    private void PunRPC_sendGraph(string jsonString)
    {
        GameSettingsSingleton.Instance.graphJsonString = jsonString;
   
    }

    [PunRPC]
    private void PunRPC_sendMesh(string objString)
    {
        GameSettingsSingleton.Instance.meshJsonString = objString;
        if (!_photonView.IsMine)
        {
            string path = Application.dataPath + "/Resources/mesh.obj";
            File.WriteAllText(path, objString);
        }
    }

    [PunRPC]
    private void PunRPC_SetNickName(string name)
    {
        gameObject.name = name;
    }

    public void OnPhotonSerializeView(PhotonStream stream, PhotonMessageInfo info)
    {
        //string path = Application.dataPath + "/Scripts/data.json";
        //string jsonString = File.ReadAllText(path);
        //Debug.Log(jsonString);

        //if (stream.IsWriting)
        //{
        //    stream.SendNext(jsonString);
        //    Debug.Log("writing");
        //}
        //else if (stream.IsReading)
        //{
        //    string jsonStringReceived = (string)stream.ReceiveNext();
        //    File.WriteAllText(path, jsonStringReceived);
        //    Debug.Log("reading");
        //}
    }

}