package pageranking;

import java.util.ArrayList;
import java.util.List;

import org.graphstream.graph.Graph;
import org.graphstream.graph.Node;

public class SimplePageRanking{

	private final Graph graph;
	private final List<Double> ranks;
	private final ElemenetSinkImpl elementSink;
	
	public SimplePageRanking(Graph graph, ElemenetSinkImpl sink) {
		this.graph = graph;
		this.graph.nodes().forEach(node -> node.setAttribute("page_rank", 1.0 / graph.getNodeCount()));
		this.ranks = new ArrayList<Double>(graph.getNodeCount());
		this.elementSink = sink;
		this.elementSink.setGraphChanged(true);
	}

	private boolean checkIfAccuracyIsAccepted(double normDistance, double precision) {
		return normDistance > precision?true:false;
	}
	
	private double computeDanglingRank(double currentDanglingRank,int outDegree, double currentPageRank) {
		return outDegree == 0?currentDanglingRank += currentPageRank:currentDanglingRank;
	}
	
	public void computeRanks() {
		double df = 0.85; //dampingFactor
		double precision = 1.0e-10;
		int i = 0, j = 0;
		double sum = 0;
		double normDistance = 0;
		
		if (!elementSink.isGraphChanged())
			return;
		
		do {
			//System.out.println("Clearing the ranks before next iteration computation..");
			ranks.clear();
			double danglingRank = 0;
			i = 0;
			while(i < graph.getNodeCount()){
				Node node = graph.getNode(i);
				int outDegree = node.getOutDegree();
				sum = 0;
				j = 0;
				while(j < node.getInDegree()) {
					Node adjacentNode = node.getEnteringEdge(j).getOpposite(node);
					//distributing the rank, uniform distribution
					sum += adjacentNode.getNumber("page_rank") / adjacentNode .getOutDegree();
					j++;
				}
				//new rank computation
				ranks.add(((1 - df) / graph.getNodeCount()) + df * sum);
				
				//if current node is dangling node then dannling node becomes the actual rank
				danglingRank = computeDanglingRank(danglingRank, outDegree, node.getNumber("page_rank"));
				i++;
			}
			danglingRank *= df / graph.getNodeCount();

			normDistance = 0;
			i = 0;
			// calculating the norm distance for each vector
			while ( i < graph.getNodeCount()) {
				normDistance += Math.abs((ranks.get(i) + danglingRank) - graph.getNode(i).getNumber("page_rank"));
				graph.getNode(i).setAttribute("page_rank", ranks.get(i) + danglingRank);
				i++;
			}
			System.out.println("ranks:"+ ranks);
		} while (checkIfAccuracyIsAccepted(normDistance, precision)); //covergence property checking precision
		elementSink.setGraphChanged(false);	
	}
}