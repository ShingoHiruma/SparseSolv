﻿
#include "Edge.h"

/*//=======================================================
  // ● コンストラクタ
  //=======================================================*/
Edge::Edge(){
	zero_edge = false;
	T_edge = false;
}
/*//=======================================================
  // ● コンストラクタ
  //=======================================================*/
Edge::Edge(int sID, int eID){
	zero_edge = false;
	T_edge = false;
	set(sID, eID);
}
/*//=======================================================
  // ● 始点・終点セッタ
  //=======================================================*/
void Edge::set(int sID, int eID){
	stat_node_id = sID; end_node_id = eID;
}
