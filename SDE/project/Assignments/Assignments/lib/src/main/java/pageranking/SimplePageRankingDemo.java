package pageranking;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import org.graphstream.algorithm.generator.DorogovtsevMendesGenerator;
import org.graphstream.graph.Graph;
import org.graphstream.graph.Node;

import crawler.WebCrawler;

/**
 * @author suresh
 * @author utkarsh
 * 
 */
public class SimplePageRankingDemo {

	/**
	 * 1. DorogovtsevMendesGenerator used for test graph data generation
	 * 2. Graph is used from graph stream library
	 * */
	public static void main(String[] args) throws InterruptedException, IOException {
		System.setProperty("org.graphstream.ui","swing");

		WebCrawler crawler = new WebCrawler();
		Graph graph = crawler.generate();
		//graph.setAttribute("ui.antialias", true);
		//graph.setAttribute("ui.stylesheet", "node {fill-color: blue; size-mode: dyn-size;} edge {fill-color:red;}");

		System.out.println("Generating the test graph using 'DorogovtsevMendesGenerator'.");
		DorogovtsevMendesGenerator generator = new DorogovtsevMendesGenerator();
		generator.setDirectedEdges(true, true);
		ElemenetSinkImpl sinkSource = new ElemenetSinkImpl(graph);
		graph.addElementSink(sinkSource);
		generator.addSink(graph);

		/*
		generator.begin();

		while (graph.getNodeCount() < 1000) {
			generator.nextEvents();
		}
		generator.end();
		*/

		/*
		 *  1.0e-10 <- dummy precision. Don't know on what basis we should take the precision
		 * */
		SimplePageRanking simplePageRankAlgo = new SimplePageRanking(graph, sinkSource);
		
		System.out.println("GENERATED GRAPH");
		System.out.println("-----------------");
		graph.forEach(each -> {
			System.out.println("node-"+each +",adjacent nodes->"+each.edges().collect(Collectors.toSet()));
		});
		
		System.out.println("");
		System.out.println("");
		System.out.println("-----------------");		
		DecimalFormat df = new DecimalFormat("#.########");
		List<String> ranks = new ArrayList<>(); 
		for (Node node : graph) {
			simplePageRankAlgo.computeRanks();
			double rank = node.getNumber("page_rank");
			String rankString = df.format(rank*100);
			ranks.add(rankString);
			//node.setAttribute("ui.class", "big, important");
			//node.setAttribute("ui.size", 50);
			node.setAttribute("ui.label",  rankString);
		}

		System.out.println("");
		System.out.println("");
		System.out.println("-----------------");
		graph.forEach(each -> {
			System.out.println("node-"+each +", rank-"+df.format(each.getNumber("page_rank")));
		});
		graph.display(true);
	}
}