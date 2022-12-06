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
        if (PhotonNetwork.NickName == "client" && _photonView.IsMine)
        {
            //send graph
            if (Input.GetKeyDown(KeyCode.Space)) // TODO send through UI buttons
            {

                SendGraph2Server();
            }
            //send boundary request
            else if (Input.GetKeyDown(KeyCode.X)) // TODO send through UI buttons
            {
                //pass
            }

        else if (PhotonNetwork.NickName == "server" && _photonView.IsMine)
            {
                //send boundary

                //send mesh

                //load up new mesh
                SendMeshData2Client();


            }
        }
    }

    private void SendGraph2Server()
    {
        //TODO update jsonString from tracked objects
        string path = Application.dataPath + "/Scripts/graph.json";
        string jsonString = File.ReadAllText(path);
        //
        _photonView.RPC("PunRPC_sendGraph", RpcTarget.AllBuffered, jsonString); //max length 32k
        Debug.Log("send out graph");
    }

    [PunRPC]
    private void PunSendBoundary()
    {

    }

    private void SendMeshData2Client()
    {
        //TODO generate mesh from rhino

        // split string into chunks
        string path = Application.dataPath + "/Resources/mesh.obj";
        string objString = File.ReadAllText(path);

        int chunkSize = 32000;
        int stringLength = objString.Length;
        List<string> objStringList = new List<string>();
        for (int i = 0; i < stringLength; i += chunkSize)
        {
            if (i + chunkSize > stringLength) chunkSize = stringLength - i;
            objStringList.Add(objString.Substring(i, chunkSize));
        }
        string[] objStringArray = objStringList.ToArray();

        Debug.Log($"string length: {stringLength}");
        Debug.Log($"string array length: {objStringArray.Length}");
        _photonView.RPC("PunPRC_sendMeshBuddle", RpcTarget.OthersBuffered, objStringArray);
    }

    [PunRPC]
    private void PunRPC_sendGraph(string jsonString)
    {
        GameSettingsSingleton.Instance.graphJsonString = jsonString;
        if (PhotonNetwork.NickName == "server" && _photonView.IsMine)
        {
            //write json
            string path = Application.dataPath + "/Resources/graph.json";
            File.WriteAllText(path, GameSettingsSingleton.Instance.graphJsonString);
        }
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
    private void PunPRC_sendMeshBuddle(string[] objStringArray)
    {
        StartCoroutine(_sendMeshBuddle(objStringArray, 0.5f));
    }

    private IEnumerator _sendMeshBuddle(string[] objStringArray, float delay)
    {
        for(int i=0; i < objStringArray.Length; i++)
        {
            GameSettingsSingleton.Instance.meshJsonString += objStringArray[i];
            yield return new WaitForSeconds(delay);
            Debug.Log($"get {i} package");
        }
        Debug.Log($"received string with length of {GameSettingsSingleton.Instance.meshJsonString.Length}");

        // received all
        if (PhotonNetwork.NickName == "client" && !_photonView.IsMine)
        {
            string path = Application.dataPath + "/Resources/mesh.obj";
            File.WriteAllText(path, GameSettingsSingleton.Instance.meshJsonString);
        }

        // call spawn mesh
        
    }
    
    [PunRPC]
    private void SpawnMeshOnReceive()
    {

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
