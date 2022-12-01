using Photon.Pun;
using Photon.Realtime;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ConnetServer : MonoBehaviourPunCallbacks
{

    void Start()
    {
        PhotonNetwork.NickName = GameSettingsSingleton.Instance.userName;
        Debug.Log("connect to server...");
        PhotonNetwork.ConnectUsingSettings();
    }

    public override void OnConnectedToMaster()
    {
        Debug.Log("server connected");
        PhotonNetwork.JoinLobby();
        StartCoroutine(CreateJoinRoom(0.1f));
    }
    public override void OnJoinedRoom()
    {
        PhotonNetwork.Instantiate("SendReceive", Vector3.zero, Quaternion.identity);
    }

    private IEnumerator CreateJoinRoom(float delay)
    {
        while (true)
        {
            yield return new WaitForSeconds(delay);
            if (PhotonNetwork.IsConnected)
            {
                RoomOptions options = new RoomOptions();
                options.MaxPlayers = 5;

                PhotonNetwork.JoinOrCreateRoom(GameSettingsSingleton.Instance.roomName, options, TypedLobby.Default);

                Debug.Log(string.Format("joined {0}", GameSettingsSingleton.Instance.roomName));

                break;
            }
        }
        
    }

}
