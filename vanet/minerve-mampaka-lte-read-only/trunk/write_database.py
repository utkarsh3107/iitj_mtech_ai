
##################################################################################
# write_database.py Python script of 				           				     #
# Development of a simulation and performance analysis platform for LTE networks #
# Project done by MINERVE MAMPAKA 					        					 #
# May 2014							        									 #
##################################################################################




#importing mysql database connectors
import mysql.connector

#importing config for csv files
import csv

#create dataset for the firt DB connection
db_url = {
            "user" : "root",
            "password" : "",
            "host" : "127.0.0.1",
            "port" : 3307,
            "database" : "project"
        }

#connect to the database		
db = mysql.connector.Connect(**db_url)

#adding cursor object for the write_db method
write_db = db.cursor()

#create python lists 
Kpi = ["Delay", "Jitter", "PacketLoss", "Throughput"]
Ue = ["1", "5", "25"]
Cache = ["OFF", "ON"]

#create function that receive kpis, number of UE and the status of the cache	
#and return a sql query
def create_tables(Kpi, Ue, Cache):


#the sql query create new tables according to the passed data	
#load the csv files into the tables
#the name of the tables and the data in the tables depends on the parsed parameters
	sql = '''
	
CREATE TABLE IF NOT EXISTS `%s%suepertraffic_cache%s` (
`%sRTP` DOUBLE NULL,
`%sCBR` DOUBLE NULL,
`%sHTTP` DOUBLE NULL,
`%sFTP` DOUBLE NULL,
`%sTotal` DOUBLE NULL
)

COMMENT='%s%suepertraffic_cache%s'
COLLATE='latin1_swedish_ci'
ENGINE=InnoDB;

TRUNCATE TABLE `project`.`%s%suepertraffic_cache%s`;

LOAD DATA INFILE 
'c:\\%s%sUEperTraffic_CACHE%s.csv' 

REPLACE INTO TABLE `project`.`%s%suepertraffic_cache%s` 
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"' ESCAPED BY '"' 
LINES TERMINATED BY '\r\n' 
(@ColVar0, @ColVar1, @ColVar2, @ColVar3, @ColVar4) 
		
SET 
`%sRTP` = REPLACE(REPLACE(@ColVar0, ' ', ''), ',', '.'), 
`%sCBR` = REPLACE(REPLACE(@ColVar1, ' ', ''), ',', '.'), 
`%sHTTP` = REPLACE(REPLACE(@ColVar2, ' ', ''), ',', '.'), 
`%sFTP` = REPLACE(REPLACE(@ColVar3, ' ', ''), ',', '.'), 
`%sTotal` = REPLACE(REPLACE(@ColVar4, ' ', ''), ',', '.');
''' \
	% (Kpi, Ue, Cache, \
	Kpi, Kpi, Kpi, Kpi, Kpi,\
	Kpi, Ue, Cache,\
	Kpi, Ue, Cache, \
	Kpi, Ue, Cache, \
	Kpi, Ue, Cache, \
	Kpi, Kpi, Kpi, Kpi, Kpi)
		
	return sql



i=0
while (i<4): #loop to create all the different testing scenarios from the project
	j=0
	while (j<3):
		k=0
		while (k<2):
			try:
				for result in write_db.execute(create_tables(Kpi[i], Ue[j], Cache[k]),multi = True):
					pass
				print "Table for %s%sUEperTraffic_CACHE%s written in database successfully\n" % (Kpi[i], Ue[j], Cache[k])
				db.commit() #execute change
			except:
				db.rollback() #rollback the database
				print "Table for %s%sUEperTraffic_CACHE%s failed to be written in database\n" % (Kpi[i], Ue[j], Cache[k])
			k+=1
		j+=1
	i+=1
	
print "Writing to project Database completed!\n\n"

#end of script


