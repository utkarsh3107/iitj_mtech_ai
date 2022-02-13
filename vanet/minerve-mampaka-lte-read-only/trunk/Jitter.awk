##################################################################################
# Jitter.awk script of 						                 #
# Development of a simulation and performance analysis platform for LTE networks #
# Project done by MINERVE MAMPAKA 					         #
# May 2014								         #
##################################################################################



BEGIN{

	#create filenames according to input parameters
	FilenameRTP="JitterRTP.txt";
	FilenameCBR="JitterCBR.txt";
	FilenameHTTP="JitterHTTP.txt";
	FilenameFTP="JitterFTP.txt";	
	FilenameTotal="JitterTotal.txt";

	#initialize variables
	for (i=0; i<=3600; i++) {
		HTTP[i]=0;
		CBR[i]=0;
		FTP[i]=0;
		RTP[i]=0;
		Total[i]=0;
	}

}

{

#save traces file results in different variables
#r  0.241408 1  0  tcp  1040 -------  1     4.0   0.0   3    6
#$1 $2       $3 $4 $5   $6      $7    $8    $9    $10   $11  $12

   action 	= $1;
   time 	= $2;
   from 	= $3;
   to 		= $4;
   type		= $5;
   pktsize 	= $6;
   flow_id 	= $8;
   src 		= $9;
   dst 		= $10;
   seq_no 	= $11;
   packet_id 	= $12;


#save time whenever the is an outgoing packet from UEs, PGW or SERVER	
#node ID 2 is the PGW, node ID 3 is the SERVER and node IDs above 3 are the UEs
	if (action == "+"&& from>1)
	{
		packet[packet_id]=time;
	}

#save the time a UE or the SERVER receive a packet
	if (action == "r" && from >2){
		if(packet[packet_id]!=0){

#the difference of current delay and previous delay gives the jitter
			Delay[flow_id,0] = time - packet[packet_id];
			if(Index>0){
				if(Delay[flow_id,0]>before[flow_id])
					Jitter[flow_id]=Jitter[flow_id]+Delay[flow_id,0]-before[flow_id];
				if(Delay[flow_id,0]<=before[flow_id])
					Jitter[flow_id]=Jitter[flow_id]+before[flow_id]-Delay[flow_id,0];
			}
			if(Index==0){
				Index=Index+1;
			}
			before[flow_id]=Delay[flow_id,0];
			Delay[flow_id,1] = Delay[flow_id,1] + 1;

		#convert the string time to integer
		TimeIndex=sprintf("%d",time);
		if (flow_id == 0) {			
			RTP[TimeIndex] = Jitter[0]/(Delay[0,1]-1);
		}
		if (flow_id == 1) {
			CBR[TimeIndex] = Jitter[1]/(Delay[1,1]-1);
		}
		if (flow_id == 2) {
			HTTP[TimeIndex] = Jitter[2]/(Delay[2,1]-1);
		}
		if (flow_id == 3) {
			FTP[TimeIndex] = Jitter[3]/(Delay[3,1]-1);
		}

		for(i=0;i<4;i++) {

		 TDelay[TimeIndex]= TDelay[TimeIndex]+Jitter[i]
		 TNum[TimeIndex]= TNum[TimeIndex]+Delay[i,1]-1;
		}

		Total[TimeIndex]=TDelay[TimeIndex]/TNum[TimeIndex]
	
			
		}

	}

}

#write the average delays in the respective files based on different traffic types
#the delays generated are expressed in second

END {      

	 	for (i=0; i<time; i++) {
		 printf("%d %f\n", i+1, RTP[i]) >> "Jitter/" FilenameRTP
		 printf("%d %f\n", i+1, CBR[i]) >> "Jitter/" FilenameCBR
		 printf("%d %f\n", i+1, HTTP[i]) >> "Jitter/" FilenameHTTP
		 printf("%d %f\n", i+1, FTP[i]) >> "Jitter/" FilenameFTP
		 printf("%d %f\n", i+1, Total[i]) >> "Jitter/" FilenameTotal
	}
}

