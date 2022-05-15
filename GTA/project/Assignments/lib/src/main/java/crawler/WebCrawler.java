package crawler;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import org.graphstream.graph.Edge;
import org.graphstream.graph.Graph;
import org.graphstream.graph.Node;
import org.graphstream.graph.implementations.SingleGraph;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Type;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class WebCrawler {

    private static final int MAX_DEPTH = 25;

    private Queue<String> queue = new LinkedList<>();

    // create set and nested list for storing links and articles
    private Map<String, Set<String>> adjacencyList;

    /**
     * Default constructor to initialize this obbject
     */
    public WebCrawler(){
        this.adjacencyList = new HashMap<>();
    }


    public void extractLinks(){
        int counter = 0;
        do{
            String url = this.queue.remove();
            this.adjacencyList.computeIfAbsent(url, key -> new HashSet<>());
            Set<String> result = new HashSet<>();
            try {
                Document doc = Jsoup.connect(url).get();
                Elements availableLinksOnPage = doc.select("a[href]");

                for(Element each : availableLinksOnPage){
                    String newURL = each.attr("abs:href").trim();
                    result.add(newURL);
                    if(!this.adjacencyList.containsKey(newURL)) {
                        this.queue.add(newURL);
                    }
                }
                this.adjacencyList.get(url).addAll(result);
            } catch (Exception ex) {
                ex.printStackTrace();
            }
            System.out.println("For Depth ["+ counter+"] parent url ["+ url +"], children : ["+result+"] ");
            counter++;
        }while(counter != MAX_DEPTH);
    }

    public Graph convertToGraph(){
        Graph graph = new SingleGraph("simple_page_ranking_graph");

        for(Map.Entry<String, Set<String>> entry: this.adjacencyList.entrySet()) {
            Node parent = graph.getNode(entry.getKey());
            if(parent == null){
                parent = graph.addNode(entry.getKey());
            }

            for(String nodeValue : entry.getValue()){
                Node child = graph.getNode(nodeValue);
                if(child == null){
                    child = graph.addNode(nodeValue);
                }

                try{
                    String id = parent + ":" + child;
                    graph.addEdge(id, parent, child);
                }catch(Exception ex){
                    System.out.println(ex.getMessage());
                }

            }
        }

        return graph;
    }

    public Map<String,Set<String>> loadFile() throws Exception{
        Gson gson = new Gson();
        String data = "";
        data = new String(Files.readAllBytes(Paths.get("/Users/utkarsh/Desktop/study/iitj_mtech_ai/GTA/project/Assignments/file1.txt")));
        Type empMapType = new TypeToken<Map<String, Set<String>>>() {}.getType();
        return gson.fromJson(data, empMapType);
    }

    public void convertToJson(String fName){
        try(FileWriter wr = new FileWriter(fName)) {
            Gson gson = new Gson();
            System.out.println(gson.toJson(this.adjacencyList));
            wr.write(gson.toJson(this.adjacencyList));
        } catch (IOException e) {
            System.err.println(e.getMessage());
        }
    }

    public Graph generate(){
        try{
            this.adjacencyList = this.loadFile();
            return this.convertToGraph();
        }catch(Exception ex){
            System.out.println(ex.getMessage());
        }

        return null;
    }

}
