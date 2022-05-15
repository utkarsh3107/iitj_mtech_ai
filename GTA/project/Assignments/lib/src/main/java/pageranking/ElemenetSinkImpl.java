package pageranking;

import org.graphstream.graph.Graph;
import org.graphstream.stream.ElementSink;

public class ElemenetSinkImpl implements ElementSink{

	private boolean graphChanged = false;
	
	private Graph graph;
	
	public static final String NODE_ATTRIBUTE_RANK = "page_rank";
	
	public static final boolean DEBUG = false;
	
	ElemenetSinkImpl(Graph graph){
		this.graph = graph;
	}
	
	public boolean isGraphChanged() {
		return graphChanged;
	}

	public void setGraphChanged(boolean graphChanged) {
		this.graphChanged = graphChanged;
	}

	@Override
	public void nodeAdded(String sourceId, long timeId, String nodeId) {
		if(DEBUG)
			System.out.println("node added.");
		// the initial rank of the new node will be 0
		graph.getNode(nodeId).setAttribute(NODE_ATTRIBUTE_RANK,
				graph.getNodeCount() == 1 ? 1.0 : 0.0);
		graphChanged = true;
	}

	@Override
	public void nodeRemoved(String sourceId, long timeId, String nodeId) {
		if(DEBUG)
			System.out.println("node removed.");
		//TODO :: Do we need node removal logic at this point in time?
		graphChanged = true;
	}

	@Override
	public void edgeAdded(String sourceId, long timeId, String edgeId, String fromNodeId, String toNodeId,
			boolean directed) {
		if(DEBUG)
			System.out.println("edge added.");
		graphChanged = true;
	}

	@Override
	public void edgeRemoved(String sourceId, long timeId, String edgeId) {
		if(DEBUG)
			System.out.println("edge removed.");
		graphChanged = true;
	}

	@Override
	public void graphCleared(String sourceId, long timeId) {
		if(DEBUG)
			System.out.println("clearing the graph.");
		graphChanged = true;
	}

	@Override
	public void stepBegins(String sourceId, long timeId, double step) {
		if(DEBUG)
			System.out.println("brgining the graph steps.");
		graphChanged = true;
	}	
}