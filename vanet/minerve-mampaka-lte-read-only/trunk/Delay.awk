##################################################################################
# Delay.awk script of 				               		         #
# Development of a simulation and performance analysis platform for LTE networks #
# Project done by MINERVE MAMPAKA 					         #
# May 2014								         #
##################################################################################



BEGIN{

	# create filenames according to input parameters
	FilenameRTP="DelayRTP.txt";
	FilenameCBR="DelayCBR.txt";
	FilenameHTTP="DelayHTTP.txt";
	FilenameFTP="DelayFTP.txt";	
	FilenameTotal="DelayTotal.txt";

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
	if (action == "+"&&((from==3)||(from==2&&flow_id==2)))
	{
		packet[packet_id]=time;
	}
	
#save the time a UE or the SERVER receive a packet
	if (action == "r" && ((from==0 && to!=1)||(to==2&&flow_id==2)))
	{

#the reception time minus the enqueue time gives a delay that we divide by the number of 
#packet transmitted in one second to find the average delay for every second
#every second average delays for different types and  total delay are saved

		if(packet[packet_id]!=0){
			Delay[flow_id,0] = Delay[flow_id,0] + time - packet[packet_id];
			Delay[flow_id,1] = Delay[flow_id,1] + 1;

			#convert the time to integers
			TimeIndex=sprintf("%d",time);
			if (flow_id == 0) {			
			RTP[TimeIndex] = Delay[0,0]/Delay[0,1];
			}
			if (flow_id == 1) {
			CBR[TimeIndex] = Delay[1,0]/Delay[1,1];
			}
			if (flow_id == 2) {
			HTTP[TimeIndex] = Delay[2,0]/Delay[2,1];
			}
			if (flow_id == 3) {
			FTP[TimeIndex] = Delay[3,0]/Delay[3,1];
			}

			for(i=0;i<4;i++) {

			 TDelay[TimeIndex]= TDelay[TimeIndex]+Delay[i,0]
			 TNum[TimeIndex]= TNum[TimeIndex]+Delay[i,1];
			}

			Total[TimeIndex]=TDelay[TimeIndex]/TNum[TimeIndex]
	
		}

          }

}

#write the average delays in the respective files based on different traffic types
#the delays generated are expressed in second

END {      
	 	for (i=0; i<time; i++) {
		 printf("%d %f\n", i+1, RTP[i]) >> "Delay/" FilenameRTP
		 printf("%d %f\n", i+1, CBR[i]) >> "Delay/" FilenameCBR
		 printf("%d %f\n", i+1, HTTP[i]) >> "Delay/" FilenameHTTP
		 printf("%d %f\n", i+1, FTP[i]) >> "Delay/" FilenameFTP
		 printf("%d %f\n", i+1, Total[i]) >> "Delay/" FilenameTotal
	}
	
}

