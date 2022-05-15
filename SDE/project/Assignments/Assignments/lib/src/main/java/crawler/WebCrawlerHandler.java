package crawler;

// //import classes available in jsoup

import com.google.gson.Gson;
import org.graphstream.graph.Graph;
import org.graphstream.graph.Node;
import org.graphstream.graph.implementations.SingleGraph;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;
//import exception, FileWriter and collection classes
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

// create ExtractArticlesExample to understand how we can extract articles
public class WebCrawlerHandler {

    // initialize MAX_DEPTH variable with final value
    private static final int MAX_DEPTH = 25;

    // create set and nested list for storing links and articles
    private final Map<String, Set<String>> adjacencyList;

    // create set and nested list for storing links and articles
    private final Set<String> visitedList;

    /**
     * Default constructor to initialize this obbject
     */
    public WebCrawlerHandler() {
        this.visitedList = new HashSet<>();
        this.adjacencyList = new LinkedHashMap<>();
    }

    /**
     * Will recursively keep on iterating over the webpage and extract links from the page. The URLs will be added to
     * the adjacencyList created where key represents primary links and value represents all links iterated from that
     * URL up to a maximum of links in the depth.
     *
     * @param URL - Input URL we need to parse
     * @param depth - Have we reached the maximum depth
     */
    public void getPageLinks(String URL, int depth) {

        try {
            // if the URL is not present in the set, we add it to the set
            adjacencyList.computeIfAbsent(URL, key -> new HashSet<>()).add(URL);

            // fetch the HTML code of the given URL by using the connect() and get() method and store the result in Document
            Document doc = Jsoup.connect(URL).get();

            // we use the select() method to parse the HTML code for extracting links of other URLs and store them into Elements
            Elements availableLinksOnPage = doc.select("a[href]");

            // increase depth
            depth++;

            // for each extracted URL, we repeat above process
            int counter = 0;
            String nextURL = URL;

            for(Element each : availableLinksOnPage){
                String newURL = each.attr("abs:href");
                if(this.adjacencyList.containsKey(newURL)){
                    System.out.println("Already traversed URL " + newURL);
                    continue;
                }
                System.out.println(">> Depth: " + depth + " [" + URL + "]");
                this.adjacencyList.computeIfAbsent(URL, eachKey -> new HashSet<>()).add(each.attr("abs:href"));
                nextURL = newURL;
                counter++;
            }

            getPageLinks(nextURL, depth);

        }
        // handle exception
        catch (IOException e) {
            // print exception messages
            System.err.println("For '" + URL + "': " + e.getMessage());
        }

        //we use the conditional statement to check whether we have already crawled the URL or not.
        // we also check whether the depth reaches to MAX_DEPTH or not
        if (adjacencyList.size() != MAX_DEPTH && !adjacencyList.containsKey(URL) && (depth < MAX_DEPTH) && (URL.startsWith("http://www.javatpoint.com") || URL.startsWith("https://www.javatpoint.com"))) {

            // use try catch block for recursive process

        }
    }

    public void convertToJson(String fName) {
        try(FileWriter wr = new FileWriter(fName)) {
            Gson gson = new Gson();
            System.out.println(gson.toJson(this.adjacencyList));
            wr.write(gson.toJson(this.adjacencyList));
        } catch (IOException e) {
            System.err.println(e.getMessage());
        }
    }

    public Graph convertToGraph(){
        Graph graph = new SingleGraph("simple_page_ranking_graph");
        this.adjacencyList.keySet().forEach(eachK ->{
            Node parentNode = graph.addNode(eachK);
            Set<String> edgesTo = this.adjacencyList.get(eachK);
            edgesTo.forEach(edgeK ->{
                Node childNode;
                if (this.adjacencyList.containsKey(edgeK)) {
                    childNode = graph.getNode(eachK);
                } else {
                    childNode = graph.addNode(edgeK);
                }
                graph.addEdge("TO",parentNode, childNode);
            });
        });

        return graph;
    }

    // main() method start
    public static void main(String[] args) {
        // create instance of the ExtractArticlesExample class
        WebCrawlerHandler obj = new WebCrawlerHandler();

        // call getPageLinks() method to get all the page links of the specified URL
        obj.getPageLinks("http://www.javatpoint.com", 0);

        obj.convertToJson("try1");
    }
}