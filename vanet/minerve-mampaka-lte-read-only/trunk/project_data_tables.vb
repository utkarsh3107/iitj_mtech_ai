
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'project_data_tables.vb script of  			                                  			'
'Development of a simulation and performance analysis platform for LTE networks	'
'Project done by MINERVE MAMPAKA 					                                    	'
'May 2014									                                                      '
'PLEASE REPLACE C:\Mampaka\PROJECT_RESULTS\ WITH YOUR FOLDERS DIRECTORY		      '
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Sub Project_Data_Tables()

Dim i, j, k As Integer
Dim Kpi(), Ue(), Cache() As Variant

Kpi = Array("Delay", "Jitter", "PacketLoss", "Throughput")
Ue = Array("1", "5", "25")
Cache = Array("OFF", "ON")

For i = 0 To 3
    For j = 0 To 2
        For k = 0 To 1
            Call Generate(Kpi(i), Ue(j), Cache(k))
        Next k
    Next j
Next i

MsgBox "Generation of DATA_TABLES completed!"
Windows("Project_Data_Tables_VB.xlsm").Activate
ActiveWindow.Close

End Sub


Sub Generate(Kpi As Variant, Ue As Variant, Cache As Variant)

    
    Windows("Project_Data_Tables_VB.xlsm").Activate
    Workbooks.OpenText FileName:= _
        "C:\Mampaka\PROJECT_RESULTS\" & Kpi & "\" & Kpi & Ue & "UEperTraffic_CACHE" & Cache & "\" & Kpi & "RTP.txt", Origin:=437, _
        StartRow:=1, DataType:=xlDelimited, TextQualifier:=xlDoubleQuote, _
        ConsecutiveDelimiter:=True, Tab:=True, Semicolon:=False, Comma:=False, _
        Space:=True, Other:=False, FieldInfo:=Array(Array(1, 1), Array(2, 1)), _
        TrailingMinusNumbers:=True
    Range("B1").Select
    Range(Selection, Selection.End(xlDown)).Select
    Selection.Copy
    Workbooks.Open FileName:="C:\Mampaka\PROJECT_RESULTS\Temp.xlsx"
    Windows("Temp.xlsx").Activate
    ActiveSheet.Paste
    Windows("Project_Data_Tables_VB.xlsm").Activate
    Workbooks.OpenText FileName:= _
        "C:\Mampaka\PROJECT_RESULTS\" & Kpi & "\" & Kpi & Ue & "UEperTraffic_CACHE" & Cache & "\" & Kpi & "CBR.txt", Origin:=437, _
        StartRow:=1, DataType:=xlDelimited, TextQualifier:=xlDoubleQuote, _
        ConsecutiveDelimiter:=True, Tab:=True, Semicolon:=False, Comma:=False, _
        Space:=True, Other:=False, FieldInfo:=Array(Array(1, 1), Array(2, 1)), _
        TrailingMinusNumbers:=True
    Range("B1").Select
    Range(Selection, Selection.End(xlDown)).Select
    Application.CutCopyMode = False
    Selection.Copy
    Windows("Temp.xlsx").Activate
    Range("B1").Select
    ActiveSheet.Paste
    Windows("Project_Data_Tables_VB.xlsm").Activate
    Workbooks.OpenText FileName:= _
        "C:\Mampaka\PROJECT_RESULTS\" & Kpi & "\" & Kpi & Ue & "UEperTraffic_CACHE" & Cache & "\" & Kpi & "HTTP.txt", Origin:=437, _
        StartRow:=1, DataType:=xlDelimited, TextQualifier:=xlDoubleQuote, _
        ConsecutiveDelimiter:=True, Tab:=True, Semicolon:=False, Comma:=False, _
        Space:=True, Other:=False, FieldInfo:=Array(Array(1, 1), Array(2, 1)), _
        TrailingMinusNumbers:=True
    Range("B1").Select
    Range(Selection, Selection.End(xlDown)).Select
    Application.CutCopyMode = False
    Selection.Copy
    Windows("Temp.xlsx").Activate
    Range("C1").Select
    ActiveSheet.Paste
    Windows(Kpi & "HTTP.txt").Activate
    Windows("Project_Data_Tables_VB.xlsm").Activate
    Workbooks.OpenText FileName:= _
        "C:\Mampaka\PROJECT_RESULTS\" & Kpi & "\" & Kpi & Ue & "UEperTraffic_CACHE" & Cache & "\" & Kpi & "FTP.txt", Origin:=437, _
        StartRow:=1, DataType:=xlDelimited, TextQualifier:=xlDoubleQuote, _
        ConsecutiveDelimiter:=True, Tab:=True, Semicolon:=False, Comma:=False, _
        Space:=True, Other:=False, FieldInfo:=Array(Array(1, 1), Array(2, 1)), _
        TrailingMinusNumbers:=True
    Range("B1").Select
    Range(Selection, Selection.End(xlDown)).Select
    Application.CutCopyMode = False
    Selection.Copy
    Windows("Temp.xlsx").Activate
    Range("D1").Select
    ActiveSheet.Paste
    Windows("Project_Data_Tables_VB.xlsm").Activate
    Workbooks.OpenText FileName:= _
        "C:\Mampaka\PROJECT_RESULTS\" & Kpi & "\" & Kpi & Ue & "UEperTraffic_CACHE" & Cache & "\" & Kpi & "Total.txt", Origin:=437, _
        StartRow:=1, DataType:=xlDelimited, TextQualifier:=xlDoubleQuote, _
        ConsecutiveDelimiter:=True, Tab:=True, Semicolon:=False, Comma:=False, _
        Space:=True, Other:=False, FieldInfo:=Array(Array(1, 1), Array(2, 1)), _
        TrailingMinusNumbers:=True
    Range("B1").Select
    Range(Selection, Selection.End(xlDown)).Select
    Application.CutCopyMode = False
    Selection.Copy
    Windows("Temp.xlsx").Activate
    Range("E1").Select
    ActiveSheet.Paste
    Application.CutCopyMode = False
    ActiveWorkbook.SaveAs FileName:="C:\Mampaka\PROJECT_RESULTS\DATA_TABLES\" & Kpi & Ue & "UEperTraffic_CACHE" & Cache, FileFormat:= _
        xlCSV, CreateBackup:=False
    ActiveWorkbook.Saved = True
    ActiveWindow.Close
    ActiveWindow.Close
    Windows(Kpi & "HTTP.txt").Activate
    ActiveWindow.Close
    Windows(Kpi & "FTP.txt").Activate
    ActiveWindow.Close
    Windows(Kpi & "CBR.txt").Activate
    ActiveWindow.Close
    Windows(Kpi & "RTP.txt").Activate
    ActiveWindow.Close
    
End Sub






