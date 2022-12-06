using Photon.Pun;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraControl : MonoBehaviour
{
    // Start is called before the first frame update
    public void SetOwnership2Client()
    {
        PhotonView pv = GetComponent<PhotonView>();
        Debug.Log(PhotonNetwork.NickName);
        if (pv.IsMine) return;

        if (PhotonNetwork.NickName == "server")
        {
            pv.RequestOwnership();
        }
    }
}
