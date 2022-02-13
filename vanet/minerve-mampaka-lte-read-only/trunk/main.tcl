##################################################################################
# main.tcl script of 						         #
# Development of a simulation and performance analysis platform for LTE networks #
# Project done by MINERVE MAMPAKA 					         #
# December 2013								         #
##################################################################################



#include other tcl files
source Parameters.tcl
source Topology.tcl
source session-rtp.tcl
source http-agent.tcl
source http-cache.tcl
source http-server.tcl


#start simulation and enable multicast 
set ns [new Simulator -multicast on]

#open file for the traces
#write the trace in the opened file
set f [open $input_(TRACES_FILENAME) w]
$ns trace-all $f   		      

#open file for the animator
#write the animator in the opened file
set nf [open $input_(ANIMATOR_NAME) w]	
$ns namtrace-all $nf	

#calculate the total number of users 
set NUMBER_OF_USERS [expr {$input_(RTP_USERS) + $input_(CBR_USERS) +$input_(HTTP_USERS) + $input_(FTP_USERS)}]

#create the nodes and the links
SetTopology 

#set different colors for different flow ID on the NAM
#flow ID 0 will be used for RTP traffic 
#flow ID 1 will be used for CBR traffic 
#flow ID 2 will be used for HTTP traffic 
#flow ID 3 will be used for FTP traffic
$ns color 0 Blue
$ns color 1 Red
$ns color 2 Yellow
$ns color 3 Green

#configure multicast protocol with route computation strategy dense mode
#and RTP multicast group
set mproto DM 				
set mrthandle [$ns mrtproto $mproto {}] 
set group [Node allocaddr]

#use session/RTP which is the agent configured in session-rtp.tcl
for { set i 0 } { $i < $input_(RTP_USERS) } {incr i} {

#configure uplink and downlink RTP sessions
set session_up($i) [new Session/RTP]	
set session_down($i) [new Session/RTP]		

#configure bandwidth for both uplink and downlink RTP sessions
$session_up($i) session_bw $input_(UP_SESSION_BANDWIDTH)		
$session_down($i) session_bw $input_(DOWN_SESSION_BANDWIDTH)		

#link RTP uplink sessions with UEs 
#link RTP downlink sessions with SERVER
$session_up($i) attach-node $UE($i)		
$session_down($i) attach-node $SERVER		

#Configure time for joining RTP group, starting RTP session and transmitting
$ns at $input_(UE_RTP_GROUP_TIME) "$session_up($i) join-group $group"
$ns at $input_(UE_RTP_START_TIME) "$session_up($i) start"
$ns at $input_(UE_RTP_TRANSMIT_TIME) "$session_up($i) transmit $input_(UP_SESSION_BANDWIDTH)"
$ns at $input_(SERVER_RTP_GROUP_TIME) "$session_down($i) join-group $group"
$ns at $input_(SERVER_RTP_START_TIME) "$session_down($i) start"
$ns at $input_(SERVER_RTP_TRANSMIT_TIME) "$session_down($i) transmit $input_(DOWN_SESSION_BANDWIDTH)"
 
}

#CBR traffic configuration without Cache Memory based on UDP protocol
for { set i $input_(RTP_USERS)} {$i < $input_(RTP_USERS) + $input_(CBR_USERS)-$input_(CBR_CACHE_USERS)} {incr i} {

#configure the SERVER as the UDP source agent without cache
#configure the UEs as the UDP sink agent without cache
set udp($i) [new Agent/UDP]		
$ns attach-agent $SERVER $udp($i)	
set null($i) [new Agent/Null]		
$ns attach-agent $UE($i) $null($i)	

#connect the source and sink UDP agents without cache
#Mark UDP for CBR traffic with Flow ID 1
$ns connect $udp($i) $null($i)		
$udp($i) set class_ 1			

#configure CBR traffic application layer without cache
set cbr($i) [new Application/Traffic/CBR] 
$cbr($i) attach-agent $udp($i)		  

#configure CBR packet size and bit rate without cache
$cbr($i) set packetSize_ $input_(CBR_PACKET_SIZE)		   
$cbr($i) set rate_ $input_(CBR_RATE)		  
$cbr($i) set random_ false		  

#configure starting time for CBR traffic generation without cache
$ns at $input_(CBR_START_TIME) "$cbr($i) start"		 
}


#CBR traffic configuration with Cache Memory based on UDP protocol
for { set i [expr {$input_(RTP_USERS)+$input_(CBR_USERS)-$input_(CBR_CACHE_USERS)}]} {$i <  $input_(RTP_USERS) + $input_(CBR_USERS)} {incr i} {

#configure the PGW as the UDP source agent with Cache
#configure the UEs as the UDP sink agent with Cache
set udp1($i) [new Agent/UDP]		
$ns attach-agent $PGW $udp1($i)	
set null1($i) [new Agent/Null]		
$ns attach-agent $UE($i) $null1($i)	

#connect the source and sink UDP agents with Cache
#Mark UDP for CBR traffic with Flow ID 1
$ns connect $udp1($i) $null1($i)		
$udp1($i) set class_ 1			

#configure CBR traffic application layer with Cache
set cbr1($i) [new Application/Traffic/CBR] 
$cbr1($i) attach-agent $udp1($i)	

#configure CBR packet size and bit rate	with Cache 
$cbr1($i) set packetSize_ $input_(CBR_PACKET_SIZE)		   
$cbr1($i) set rate_ $input_(CBR_RATE)		 
$cbr1($i) set random_ false	

#configure starting time for CBR traffic generation with Cache
$ns at $input_(CBR_START_TIME) "$cbr1($i) start"		 
}

#create HTTP session: HTTP is used as interactive traffic

#enable session routing for this simulation
#open log file to save http trace
$ns rtproto Session ; 
set log [open "http.log" w] 

#HTTP page configuration
set pgp [new PagePool/Math]
set tmp [new RandomVariable/Constant] 
$tmp set val_ $input_(AVERAGE_PAGE_SIZE) 
$pgp ranvar-size $tmp
set tmp [new RandomVariable/Exponential] 
$tmp set avg_ $input_(AVERAGE_PAGE_AGE) 
$pgp ranvar-age $tmp

#HTTP server configuration
set s [new Http/Server $ns $SERVER]
$s set-page-generator $pgp
$s log $log

#HTTP Cache configuration with the PGW as the Cache
set cache [new Http/Cache $ns $PGW]
$cache log $log

#configure UEs as HTTP clients
for { set i [expr {$input_(RTP_USERS) + $input_(CBR_USERS)}]} {$i<$input_(RTP_USERS) + $input_(CBR_USERS) +$input_(HTTP_USERS)} {incr i} {
set c($i) [new Http/Client $ns $UE($i)]

#configure HTTP request parameters
set ctmp($i) [new RandomVariable/Exponential] 
$ctmp($i) set avg_ $input_(AVERAGE_REQ_INTERVAL) 
$c($i) set-interval-generator $ctmp($i)
$c($i) set-page-generator $pgp
$c($i) log $log

}

#start HTTP session 
$ns at $input_(HTTP_START_TIME) "start-connection"

#configuration of HTTP session with Cache Memory
proc start-connection {} {

#declare process variables
global ns s cache c input_

$cache connect $s
 for { set i [expr {$input_(RTP_USERS) + $input_(CBR_USERS)}]} {$i<$input_(RTP_USERS) + $input_(CBR_USERS) +$input_(HTTP_USERS)} {incr i} {
	$c($i) connect $cache
	$c($i) start-session $cache $s

 }
}

#FTP traffic configuration without Cache Memory based on TCP protocol
for { set i [expr {$input_(RTP_USERS) + $input_(CBR_USERS) +$input_(HTTP_USERS)}]} {$i < $NUMBER_OF_USERS-$input_(FTP_CACHE_USERS)} {incr i} {

#configure the SERVER as the TCP source agent without Cache
#configure the UEs as the TCP sink agent without Cache
set tcp($i) [new Agent/TCP]		
$ns attach-agent $SERVER $tcp($i)	
set sink($i) [new Agent/TCPSink]
$ns attach-agent $UE($i) $sink($i)	

#connect the source and sink TCP agents without Cache
#Mark TCP for FTP traffic with Flow ID 3	
$ns connect $tcp($i) $sink($i)		
$tcp($i) set class_ 3			

#configure FTP packet size without Cache
$tcp($i) set packetSize_ $input_(FTP_PACKET_SIZE)		

#configure FTP traffic application layer without Cache
set ftp($i) [new Application/FTP]	
$ftp($i) attach-agent $tcp($i)		

#configure starting time for FTP traffic generation without Cache
$ns at $input_(FTP_START_TIME) "$ftp($i) start"		

}

#FTP traffic configuration with Cache Memory based on TCP protocol
for { set i [expr {$NUMBER_OF_USERS-$input_(FTP_CACHE_USERS)}]} {$i < $NUMBER_OF_USERS} {incr i} {

#configure the PGW as the TCP source agent with Cache
#configure the UEs as the TCP sink agent with Cache
set tcp1($i) [new Agent/TCP]		
$ns attach-agent $PGW $tcp1($i)	
set sink1($i) [new Agent/TCPSink]	
$ns attach-agent $UE($i) $sink1($i)

#connect the source and sink TCP agents with Cache
#Mark TCP for FTP traffic with Flow ID 3		
$ns connect $tcp1($i) $sink1($i)		
$tcp1($i) set class_ 3			

#configure FTP packet size with Cache
$tcp1($i) set packetSize_ $input_(FTP_PACKET_SIZE)		

#configure FTP traffic application layer with Cache
set ftp1($i) [new Application/FTP]	 
$ftp1($i) attach-agent $tcp1($i)

#configure starting time for FTP traffic generation with Cache		
$ns at $input_(FTP_START_TIME) "$ftp1($i) start"	

}

#configure end of simulation based on user input time
$ns at $input_(TIME_SIMULATION) "finish"			

#configure the end of simulation process
proc finish {} {

#declare process variables
global ns f nf input_ log

#flush the trace and log
#close both the trace, log and the NAM files
$ns flush-trace
close $f		
close $nf

#execute the NAM file before closing the simulation
#exec nam $input_(ANIMATOR_NAME) &	

#terminate successfully the process
exit 0			
}

#run the simulation
$ns run			

#end of program


