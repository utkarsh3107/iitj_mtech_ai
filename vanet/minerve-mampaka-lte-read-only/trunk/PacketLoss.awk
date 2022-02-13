##################################################################################
# PacketLoss.awk script of 						         #
# Development of a simulation and performance analysis platform for LTE networks #
# Project done by MINERVE MAMPAKA 					         #
# May 2014								         #
##################################################################################



BEGIN {
	
	#create filenames according to input parameters
	FilenameRTP="PacketLossRTP.txt";
	FilenameCBR="PacketLossCBR.txt";
	FilenameHTTP="PacketLossHTTP.txt";
	FilenameFTP="PacketLossFTP.txt";	
	FilenameTotal="PacketLossTotal.txt";

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


#calculate the size of the packet lost whenever the event is a packet dropped
#every second sum of dropped packet giving PacketLoss for different types and  total PacketLoss are saved
        if (action=="d") {
		TimeIndex=sprintf("%d",time);
		if (flow_id == 0) {			
			RTP[TimeIndex] = RTP[TimeIndex] + pktsize;
		}
		if (flow_id == 1) {
			CBR[TimeIndex] = CBR[TimeIndex] + pktsize;
		}
		if (flow_id == 2) {
			HTTP[TimeIndex] = HTTP[TimeIndex] + pktsize;
		}
		if (flow_id == 3) {
			FTP[TimeIndex] = FTP[TimeIndex] + pktsize;
		}
		Total[TimeIndex] = Total[TimeIndex] + pktsize;
	}
}	

#write the PacketLoss in the respective files based on different traffic types
#the PacketLoss generated are expressed in Mbit/s

END {
 	for (i=0; i<time; i++) {
		printf("%d %f\n", i+1, RTP[i]*8/(1024*1024)) >> "PacketLoss/" FilenameRTP
		printf("%d %f\n", i+1, CBR[i]*8/(1024*1024)) >> "PacketLoss/" FilenameCBR
		printf("%d %f\n", i+1, HTTP[i]*8/(1024*1024)) >> "PacketLoss/" FilenameHTTP
		printf("%d %f\n", i+1, FTP[i]*8/(1024*1024)) >> "PacketLoss/" FilenameFTP
		printf("%d %f\n", i+1, Total[i]*8/(1024*1024)) >> "PacketLoss/" FilenameTotal
	}	
}

