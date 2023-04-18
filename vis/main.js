var nodes, edges, network;

var nodes = new vis.DataSet(data["nodes"]);
var edges = new vis.DataSet(data["edges"]);

var container = document.getElementById("mynetwork");
var options = {};
var network = new vis.Network(container, {nodes: nodes, edges: edges}, options);

var slider = document.getElementById("slider");
slider.oninput = function() {
  edges.forEach(x => {
    if (x.ts == this.value) {
      edges.update({id: x.id, color: 'red', width: 5})
    } else {
      edges.update({id: x.id, color: 'blue', width: 1})
    }
  })
}
