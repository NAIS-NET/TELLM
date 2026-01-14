import os
import torch
import time
from omegaconf import OmegaConf
from src.model.tellm.pipline_tellm import TeLLMInferencePipline
from tellm_main import TEDatasetWithinCluster
from src.utils import print_config, logging, display_result  
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, send_file, request, render_template_string, jsonify
import os
from matplotlib.colors import LinearSegmentedColormap, to_hex
import base64
import json
from collections import deque

# Apply rounding to any remaining float values we may have missed
def round_floats_recursive(obj, decimal_places=3):
    """Recursively round all floating point values in a nested data structure"""
    if isinstance(obj, float):
        return round(obj, decimal_places)
    elif isinstance(obj, dict):
        return {k: round_floats_recursive(v, decimal_places) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats_recursive(i, decimal_places) for i in obj]
    else:
        return obj


app = Flask(__name__)

# Node location data
node_locations = [
    {"name": "ATLA-M5", "city": "Atlanta_GA", "latitude": 31.750000, "longitude": -81.383300},
    {"name": "ATLAng", "city": "Atlanta_GA", "latitude": 33.750000, "longitude": -84.383300},
    {"name": "CHINng", "city": "Chicago_IL", "latitude": 43.833300, "longitude": -87.616700},
    {"name": "DNVRng", "city": "Denver_CO", "latitude": 40.750000, "longitude": -105.000000},
    {"name": "HSTNng", "city": "Houston_TX", "latitude": 29.770031, "longitude": -95.517364},
    {"name": "IPLSng", "city": "Indianapolis_IN", "latitude": 39.780622, "longitude": -86.159535},
    {"name": "KSCYng", "city": "Kansas_City_MO", "latitude": 38.961694, "longitude": -96.596704},
    {"name": "LOSAng", "city": "Los_Angeles_CA", "latitude": 34.050000, "longitude": -118.250000},
    {"name": "NYCMng", "city": "New_York_NY", "latitude": 40.783300, "longitude": -73.966700},
    {"name": "SNVAng", "city": "Sunnyvale_CA", "latitude": 37.38575, "longitude": -122.02553},
    {"name": "STTLng", "city": "Seattle_WA", "latitude": 47.600000, "longitude": -122.300000},
    {"name": "WASHng", "city": "Washington_DC", "latitude": 38.897303, "longitude": -77.026842}
]

# Initialize model and dataset
ckpt_path = 'outputs/tellm-instruct-new-loop5'
MODEL_ID = os.path.join(ckpt_path, "pretrained")
dataset="abilene"
gpu_id = 1
num_paths_per_pair=4
num_for_loops = 5

pipe = TeLLMInferencePipline(MODEL_ID, device=f"cuda:{gpu_id}", torch_dtype=torch.float32)
policy_embeds=[0.0, 0.0, 0.0]

abilene_config = {
    "topo": "abilene",
    "weight": None,
    "metric": "MLU",
    "num_paths_per_pair": num_paths_per_pair,
    "framework": "harp",
    "failure_id": None,
    "dynamic": True,
    "mode": "train",
    "pred": False,
    "hist_len": 12
}

dataset_cfg = OmegaConf.create(abilene_config)
test_dataset = TEDatasetWithinCluster(dataset_cfg, cluster=0, start=0, end=100)
num_samples = len(test_dataset)

# Store results for each sample
results = {}
sample_idx = 0
out_queue = deque(maxlen=12)

# Generate base network topology once
base_network_html = None

def generate_base_network():
    """Generate the base network topology without utilization coloring"""
    global base_network_html
    
    G = test_dataset.snapshot.graph
    
    # Create a pyvis network
    net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black", notebook=False)
    net.toggle_physics(False)
    
    # Scale coordinates
    min_lat = min(loc["latitude"] for loc in node_locations)
    max_lat = max(loc["latitude"] for loc in node_locations)
    min_lon = min(loc["longitude"] for loc in node_locations)
    max_lon = max(loc["longitude"] for loc in node_locations)
    
    # Read and encode router SVG icon
    router_icon_path = "assets/router.svg"
    with open(router_icon_path, 'rb') as f:
        svg_data = f.read()
    router_icon_base64 = base64.b64encode(svg_data).decode('utf-8')
    router_icon_uri = f"data:image/svg+xml;base64,{router_icon_base64}"
    
    # Add nodes with embedded icon
    for i, node in enumerate(G.nodes()):
        loc = node_locations[node]
        x = (loc["longitude"] - min_lon) / (max_lon - min_lon) * 1000 - 500
        y = -((loc["latitude"] - min_lat) / (max_lat - min_lat) * 500 - 250)
        
        net.add_node(node, 
                    x=x, y=y,
                    title=f"Node {node}: {loc['city']}",
                    shape='image',
                    image=router_icon_uri,
                    size=25)
    
    # Add edges with default gray color
    edge_list = []
    for source, target, data in G.edges(data=True):
        capacity = float(data['capacity'])
        width = 1 + capacity / 3
        
        net.add_edge(source, target, 
                   id=f"{source}-{target}",
                   width=2*width, 
                   title=f"Capacity: {capacity} Gbps",
                   color="#808080")  # Default gray color
        
        edge_list.append({"from": source, "to": target, "id": f"{source}-{target}"})
    
    # Configure options
    net.set_options("""
    {
      "physics": {
        "enabled": false,
        "stabilization": false
      },
      "interaction": {
        "hover": true,
        "navigationButtons": false,
        "multiselect": true,
        "dragNodes": true
      },
      "edges": {
        "smooth": false,  
        "arrows": {
          "to": {
            "enabled": false
          },
          "from": {
            "enabled": false
          }
        }
      }
    }
    """)
    
    # Generate the html file
    static_dir = "assets"
    os.makedirs(static_dir, exist_ok=True)
    output_path = os.path.join(static_dir, "abilene_network_base.html")
    net.save_graph(output_path)
    
    # Read and modify the HTML to add dynamic color updating capability
    with open(output_path, 'r') as file:
        base_network_html = file.read()
    
    # Store edge list for reference
    return edge_list

# Generate base network on startup
edge_list = generate_base_network()

@app.route('/')
def index():
    global sample_idx, out_queue

    sample_idx = request.args.get('sample', default=0, type=int)
    auto_mode = request.args.get('auto', default="false")
    speed = request.args.get('speed', default=200, type=int)
    transition_duration = request.args.get('transition', default=100, type=int)
    
    if sample_idx >= num_samples:
        sample_idx = 0
    
    # Always run inference for this sample
    out = pipe(test_dataset[sample_idx], policy_embeds=policy_embeds, num_for_loops=num_for_loops, 
                num_paths_per_pair=num_paths_per_pair, return_loss=True)
    results[sample_idx] = out
    out_queue.append(out)
    
    # Get edge utilization data
    edge_colors = get_edge_colors(sample_idx, out)
    
    # Create the HTML page with embedded network and dynamic updating
    # Replace the policy panel section in the HTML content with this updated version:

    html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Network Topology Visualization</title>
            <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
            <style>
                #mynetwork {{
                    width: 100%;
                    height: 500px;
                    border: 1px solid lightgray;
                }}
                .control-panel {{
                    padding: 15px;
                    background-color: #f5f5f5;
                    margin-bottom: 20px;
                }}
                .button-green {{
                    background-color: #4CAF50;
                    color: white;
                    padding: 8px 15px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                }}
                .button-red {{
                    background-color: #f44336;
                    color: white;
                    padding: 8px 15px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                }}
                .button-blue {{
                    background-color: #2196F3;
                    color: white;
                    padding: 8px 15px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                }}
                .slider-container {{
                    display: inline-block;
                    margin-left: 20px;
                }}
                .slider {{
                    width: 100px;
                    vertical-align: middle;
                }}
                .policy-panel {{
                    margin-top: 15px;
                    padding: 15px;
                    background-color: #e0e0e0;
                    border-radius: 5px;
                }}
                .policy-slider-group {{
                    margin: 15px 0;
                }}
                .policy-slider {{
                    width: 150;
                    vertical-align: middle;
                }}
                .policy-value {{
                    display: inline-block;
                    width: 60px;
                    font-weight: regular;
                    color: #2196F3;
                    text-align: middle;
                    margin-left: 10px;
                }}
                .policy-label {{
                    display: inline-block;
                    width: 110px;
                    font-weight: regular;
                }}
                .status-message {{
                    margin-top: 10px;
                    padding: 10px;
                    border-radius: 4px;
                    display: none;
                }}
                .status-success {{
                    background-color: #d4edda;
                    color: #155724;
                    border: 1px solid #c3e6cb;
                }}
                .status-error {{
                    background-color: #f8d7da;
                    color: #721c24;
                    border: 1px solid #f5c6cb;
                }}
                .policy-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 15px;
                }}
                .current-embeddings {{
                    background-color: #fff;
                    padding: 8px 12px;
                    border-radius: 4px;
                    font-family: monospace;
                    font-size: 14px;
                }}
            </style>
        </head>
        <body>
            <div class="control-panel">
                <form action="/" method="get" id="sampleForm">
                    <label for="sample">Select Traffic Matrix Sample (0-{num_samples-1}):</label>
                    <input type="number" id="sample" name="sample" min="0" max="{num_samples-1}" value="{sample_idx}">
                    <button type="submit">Load</button>
                    
                    <div style="margin-top: 10px; display: flex; align-items: center;">
                        <button type="button" id="startIteration" class="button-green" style="{'' if auto_mode == 'false' else 'display: none;'}">Start Auto-Iteration</button>
                        <button type="button" id="stopIteration" class="button-red" style="{'' if auto_mode != 'false' else 'display: none;'}">Stop Auto-Iteration</button>
                        
                        <div class="slider-container">
                            <label for="iterationSpeed">Speed: <span id="speedValue">{speed}</span>ms</label>
                            <input type="range" id="iterationSpeed" class="slider" min="200" max="2000" step="100" value="{speed}">
                        </div>
                        
                        <div class="slider-container">
                            <label for="transitionDuration">Transition: <span id="transitionValue">{transition_duration}</span>ms</label>
                            <input type="range" id="transitionDuration" class="slider" min="100" max="2000" step="100" value="{transition_duration}">
                        </div>
                    </div>
                </form>
                
                <div class="policy-panel">
                    <div class="policy-header">
                        <h4 style="margin: 0;">Policy Embeddings Configuration</h4>
                        <div class="current-embeddings">
                            Current: [<span id="currentPolicyEmbeds">{', '.join(map(lambda x: f'{x:.2f}', policy_embeds))}</span>]
                        </div>
                    </div>
                    
                    <div class="policy-slider-group">
                        <span class="policy-label">Embedding 1:</span>
                        <input type="range" id="policy0" class="policy-slider" 
                            min="0" max="1" step="0.01" value="{policy_embeds[0]}">
                        <span class="policy-value" id="policy0Value">{policy_embeds[0]:.2f}</span>
                    </div>
                    
                    <div class="policy-slider-group">
                        <span class="policy-label">Embedding 2:</span>
                        <input type="range" id="policy1" class="policy-slider" 
                            min="0" max="1" step="0.01" value="{policy_embeds[1]}">
                        <span class="policy-value" id="policy1Value">{policy_embeds[1]:.2f}</span>
                    </div>
                    
                    <div class="policy-slider-group">
                        <span class="policy-label">Embedding 3:</span>
                        <input type="range" id="policy2" class="policy-slider" 
                            min="0" max="1" step="0.01" value="{policy_embeds[2]}">
                        <span class="policy-value" id="policy2Value">{policy_embeds[2]:.2f}</span>
                    </div>
                    
                    <div style="margin-top: 15px;">
                        <button type="button" id="resetPolicy" class="button-blue">Reset to [0, 0, 0]</button>
                        <button type="button" id="clearCache" class="button-red" style="margin-left: 10px;">Clear Cache & Recompute</button> 
                    </div>
                    <div id="statusMessage" class="status-message"></div>
                </div>
                
                <div style="margin-top: 10px;">
                    <p><strong>Current Sample:</strong> <span id="currentSample">{sample_idx}</span></p>
                    <p><strong>Normalized Maximum Link Utilization (Norm MLU):</strong> <span id="normMLU">{out.loss.norm_mlu.item():.4f}</span></p>
                    <p><strong>Maximum Sensitivity:</strong> <span id="maxSensitivity">{out.loss.max_sensitivity.item():.4f}</span></p>
                    <p><strong>Cost</strong> <span id="cost">{calculate_cost(out, test_dataset[sample_idx]):.4f}</span></p>
                </div>
                <div style="margin-top: 10px;">
                    <strong>Normalized Utilization:</strong>
                    <div style="margin-top: 5px;">
                        <div style="width: 300px; height: 20px; background: linear-gradient(to right, green, yellow, red);"></div>
                        <div style="display: flex; width: 300px;">
                            <span style="flex: 1; text-align: left;">0%</span>
                            <span style="flex: 1; text-align: center;">50%</span>
                            <span style="flex: 1; text-align: right;">100%</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div id="mynetwork"></div>
            
            <script type="text/javascript">
                // Network data (initialized once)
                var nodes = null;
                var edges = null;
                var network = null;
                var currentEdgeColors = {{}};
                var targetEdgeColors = {{}};
                var animationFrameId = null;
                var transitionStartTime = null;
                var transitionDuration = {transition_duration};
                
                // Debounce function to limit API calls
                function debounce(func, wait) {{
                    let timeout;
                    return function executedFunction(...args) {{
                        const later = () => {{
                            clearTimeout(timeout);
                            func(...args);
                        }};
                        clearTimeout(timeout);
                        timeout = setTimeout(later, wait);
                    }};
                }}
                
                // Initialize network
                function initNetwork() {{
                    // Load base network from saved HTML
                    fetch('/get_network_data')
                        .then(response => response.json())
                        .then(data => {{
                            nodes = new vis.DataSet(data.nodes);
                            edges = new vis.DataSet(data.edges);
                            
                            var container = document.getElementById('mynetwork');
                            var networkData = {{
                                nodes: nodes,
                                edges: edges
                            }};
                            
                            var options = data.options;
                            network = new vis.Network(container, networkData, options);
                            
                            // Apply initial colors
                            var initialColors = {json.dumps(edge_colors)};
                            for (var edgeId in initialColors) {{
                                currentEdgeColors[edgeId] = initialColors[edgeId].color;
                            }}
                            updateEdgeColors(initialColors);
                        }});
                }}
                
                // Parse hex color to RGB
                function hexToRgb(hex) {{
                    var result = /^#?([a-f\d]{{2}})([a-f\d]{{2}})([a-f\d]{{2}})$/i.exec(hex);
                    return result ? {{
                        r: parseInt(result[1], 16),
                        g: parseInt(result[2], 16),
                        b: parseInt(result[3], 16)
                    }} : null;
                }}
                
                // Convert RGB to hex
                function rgbToHex(r, g, b) {{
                    return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
                }}
                
                // Interpolate between two colors
                function interpolateColor(color1, color2, factor) {{
                    var c1 = hexToRgb(color1);
                    var c2 = hexToRgb(color2);
                    
                    if (!c1 || !c2) return color1;
                    
                    var r = Math.round(c1.r + (c2.r - c1.r) * factor);
                    var g = Math.round(c1.g + (c2.g - c1.g) * factor);
                    var b = Math.round(c1.b + (c2.b - c1.b) * factor);
                    
                    return rgbToHex(r, g, b);
                }}
                
                // Animate color transition
                function animateColorTransition() {{
                    if (!transitionStartTime) {{
                        transitionStartTime = Date.now();
                    }}
                    
                    var elapsed = Date.now() - transitionStartTime;
                    var progress = Math.min(elapsed / transitionDuration, 1);
                    
                    // Use easing function for smoother transition
                    var easedProgress = 1 - Math.pow(1 - progress, 3); // Cubic ease-out
                    
                    var updates = [];
                    for (var edgeId in targetEdgeColors) {{
                        var currentColor = currentEdgeColors[edgeId] || '#808080';
                        var targetColor = targetEdgeColors[edgeId].color;
                        var interpolatedColor = interpolateColor(currentColor, targetColor, easedProgress);
                        
                        updates.push({{
                            id: edgeId,
                            color: interpolatedColor,
                            title: targetEdgeColors[edgeId].title
                        }});
                    }}
                    
                    edges.update(updates);
                    
                    if (progress < 1) {{
                        animationFrameId = requestAnimationFrame(animateColorTransition);
                    }} else {{
                        // Animation complete, update current colors
                        for (var edgeId in targetEdgeColors) {{
                            currentEdgeColors[edgeId] = targetEdgeColors[edgeId].color;
                        }}
                        transitionStartTime = null;
                    }}
                }}
                
                // Update edge colors with animation
                function updateEdgeColorsAnimated(edgeColors) {{
                    // Cancel any ongoing animation
                    if (animationFrameId) {{
                        cancelAnimationFrame(animationFrameId);
                        animationFrameId = null;
                    }}
                    
                    // Set new target colors
                    targetEdgeColors = edgeColors;
                    transitionStartTime = null;
                    
                    // Start animation
                    animateColorTransition();
                }}
                
                // Update edge colors immediately (without animation)
                function updateEdgeColors(edgeColors) {{
                    if (edges) {{
                        var updates = [];
                        for (var edgeId in edgeColors) {{
                            updates.push({{
                                id: edgeId,
                                color: edgeColors[edgeId].color,
                                title: edgeColors[edgeId].title
                            }});
                            currentEdgeColors[edgeId] = edgeColors[edgeId].color;
                        }}
                        edges.update(updates);
                    }}
                }}
                
                // Update sample via AJAX with animated transition
                function updateSample(sampleIdx) {{
                    fetch('/get_sample_data?sample=' + sampleIdx)
                        .then(response => response.json())
                        .then(data => {{
                            // Update displayed values
                            document.getElementById('currentSample').textContent = sampleIdx;
                            document.getElementById('normMLU').textContent = data.norm_mlu.toFixed(4);
                            document.getElementById('maxSensitivity').textContent = data.max_sensitivity.toFixed(4);
                            document.getElementById('cost').textContent = data.cost.toFixed(4);
                            
                            // Update edge colors with animation
                            updateEdgeColorsAnimated(data.edge_colors);
                        }});
                }}
                
                // Show status message
                function showStatus(message, isSuccess) {{
                    var statusDiv = document.getElementById('statusMessage');
                    statusDiv.textContent = message;
                    statusDiv.className = 'status-message ' + (isSuccess ? 'status-success' : 'status-error');
                    statusDiv.style.display = 'block';
                    
                    // Hide after 3 seconds
                    setTimeout(function() {{
                        statusDiv.style.display = 'none';
                    }}, 3000);
                }}
                // Track if user is actively interacting with sliders
                let isUserDragging = false;
                
                function pullPolicyEmbeddings() {{
                    // Don't update if user is actively dragging
                    if (isUserDragging) return;
                    
                    fetch('/api/policy')
                        .then(response => response.json())
                        .then(data => {{
                            if (data.success) {{
                                // Update the display
                                document.getElementById('currentPolicyEmbeds').textContent = 
                                    data.policy_embeds.map(v => v.toFixed(2)).join(', ');
                                
                                // Check if values have changed before updating sliders
                                const currentValues = [
                                    parseFloat(document.getElementById('policy0').value),
                                    parseFloat(document.getElementById('policy1').value),
                                    parseFloat(document.getElementById('policy2').value)
                                ];
                                
                                const hasChanged = data.policy_embeds.some((val, idx) => 
                                    Math.abs(val - currentValues[idx]) > 0.001
                                );
                                
                                if (hasChanged) {{
                                    // Update the slider positions and their displayed values
                                    document.getElementById('policy0').value = data.policy_embeds[0];
                                    document.getElementById('policy1').value = data.policy_embeds[1];
                                    document.getElementById('policy2').value = data.policy_embeds[2];
                                    
                                    // Update the value displays next to sliders
                                    document.getElementById('policy0Value').textContent = data.policy_embeds[0].toFixed(2);
                                    document.getElementById('policy1Value').textContent = data.policy_embeds[1].toFixed(2);
                                    document.getElementById('policy2Value').textContent = data.policy_embeds[2].toFixed(2);
                                    
                                    // If values changed externally, reload the current sample
                                    var currentSample = parseInt(document.getElementById('currentSample').textContent);
                                    updateSample(currentSample);
                                }}
                            }}
                        }})
                        .catch(error => {{
                            console.error('Error fetching policy embeddings:', error);
                        }});
                }}
                
                // Add mousedown/mouseup listeners to track dragging
                ['policy0', 'policy1', 'policy2'].forEach(id => {{
                    document.getElementById(id).addEventListener('mousedown', () => {{
                        isUserDragging = true;
                    }});
                    
                    document.getElementById(id).addEventListener('mouseup', () => {{
                        isUserDragging = false;
                    }});
                    
                    // Also handle touch events for mobile
                    document.getElementById(id).addEventListener('touchstart', () => {{
                        isUserDragging = true;
                    }});
                    
                    document.getElementById(id).addEventListener('touchend', () => {{
                        isUserDragging = false;
                    }});
                }});

                
                // Poll for policy updates every 2 seconds
                setInterval(pullPolicyEmbeddings, 2000);
                
                // Initial pull on page load
                document.addEventListener('DOMContentLoaded', pullPolicyEmbeddings);


                
                // Update policy embeddings with debounce
                const updatePolicyEmbeddings = debounce(function() {{
                    var policy0 = parseFloat(document.getElementById('policy0').value);
                    var policy1 = parseFloat(document.getElementById('policy1').value);
                    var policy2 = parseFloat(document.getElementById('policy2').value);
                    
                    fetch('/update_policy', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                        }},
                        body: JSON.stringify({{
                            policy_embeds: [policy0, policy1, policy2]
                        }})
                    }})
                    .then(response => response.json())
                    .then(data => {{
                        if (data.success) {{
                            document.getElementById('currentPolicyEmbeds').textContent = 
                                data.policy_embeds.map(v => v.toFixed(2)).join(', ');
                            showStatus('Policy embeddings updated!', true);
                            
                            // Reload current sample with new policy embeddings
                            var currentSample = parseInt(document.getElementById('currentSample').textContent);
                            updateSample(currentSample);
                        }} else {{
                            showStatus('Error: ' + data.error, false);
                        }}
                    }})
                    .catch(error => {{
                        showStatus('Error updating policy embeddings: ' + error, false);
                    }});
                }}, 300); // Debounce delay of 300ms
                
                // Add event listeners for policy sliders
                document.getElementById('policy0').addEventListener('input', function() {{
                    document.getElementById('policy0Value').textContent = parseFloat(this.value).toFixed(2);
                    updatePolicyEmbeddings();
                }});
                
                document.getElementById('policy1').addEventListener('input', function() {{
                    document.getElementById('policy1Value').textContent = parseFloat(this.value).toFixed(2);
                    updatePolicyEmbeddings();
                }});
                
                document.getElementById('policy2').addEventListener('input', function() {{
                    document.getElementById('policy2Value').textContent = parseFloat(this.value).toFixed(2);
                    updatePolicyEmbeddings();
                }});
                
                // Reset policy embeddings
                document.getElementById('resetPolicy').addEventListener('click', function() {{
                    // Reset sliders to 0
                    document.getElementById('policy0').value = 0;
                    document.getElementById('policy1').value = 0;
                    document.getElementById('policy2').value = 0;
                    
                    // Update displayed values
                    document.getElementById('policy0Value').textContent = '0.00';
                    document.getElementById('policy1Value').textContent = '0.00';
                    document.getElementById('policy2Value').textContent = '0.00';
                    
                    // Send update to server
                    updatePolicyEmbeddings();
                }});
                
                // Clear cache and recompute
                document.getElementById('clearCache').addEventListener('click', function() {{
                    if (!confirm('This will clear all cached results and recompute with current policy embeddings. Continue?')) {{
                        return;
                    }}
                    
                    fetch('/clear_cache', {{
                        method: 'POST'
                    }})
                    .then(response => response.json())
                    .then(data => {{
                        if (data.success) {{
                            showStatus('Cache cleared successfully! Results will be recomputed.', true);
                            
                            // Reload current sample
                            var currentSample = parseInt(document.getElementById('currentSample').textContent);
                            updateSample(currentSample);
                        }} else {{
                            showStatus('Error: ' + data.error, false);
                        }}
                    }})
                    .catch(error => {{
                        showStatus('Error clearing cache: ' + error, false);
                    }});
                }});
                
                // Auto-iteration functionality
                let currentSample = {sample_idx};
                const maxSample = {num_samples-1};
                let autoMode = "{auto_mode}";
                let speed = {speed};
                let autoInterval = null;
                
                // Update speed display
                document.getElementById('iterationSpeed').addEventListener('input', function() {{
                    speed = this.value;
                    document.getElementById('speedValue').textContent = speed;
                    
                    // If auto-iteration is running, restart with new speed
                    if (autoInterval) {{
                        clearInterval(autoInterval);
                        autoInterval = setInterval(function() {{
                            currentSample = (currentSample + 1) % (maxSample + 1);
                            updateSample(currentSample);
                            document.getElementById('sample').value = currentSample;
                        }}, speed);
                    }}
                }});
                
                // Update transition duration display
                document.getElementById('transitionDuration').addEventListener('input', function() {{
                    transitionDuration = parseInt(this.value);
                    document.getElementById('transitionValue').textContent = transitionDuration;
                }});
                
                document.getElementById('startIteration').addEventListener('click', function() {{
                    autoMode = "true";
                    document.getElementById('startIteration').style.display = 'none';
                    document.getElementById('stopIteration').style.display = 'block';
                    
                    // Start auto-iteration
                    autoInterval = setInterval(function() {{
                        currentSample = (currentSample + 1) % (maxSample + 1);
                        updateSample(currentSample);
                        document.getElementById('sample').value = currentSample;
                    }}, speed);
                }});
                
                document.getElementById('stopIteration').addEventListener('click', function() {{
                    autoMode = "false";
                    document.getElementById('startIteration').style.display = 'block';
                    document.getElementById('stopIteration').style.display = 'none';
                    
                    // Stop auto-iteration
                    if (autoInterval) {{
                        clearInterval(autoInterval);
                        autoInterval = null;
                    }}
                }});
                
                // Initialize network on page load
                initNetwork();
                
                // Start auto-iteration if in auto mode
                if (autoMode === "true") {{
                    document.getElementById('startIteration').style.display = 'none';
                    document.getElementById('stopIteration').style.display = 'block';
                    
                    autoInterval = setInterval(function() {{
                        currentSample = (currentSample + 1) % (maxSample + 1);
                        updateSample(currentSample);
                        document.getElementById('sample').value = currentSample;
                    }}, speed);
                }}
            </script>
        </body>
        </html>
        """

    
    return html_content

def calculate_cost(out, sample_data):
    split_ratios = out.split_ratios.squeeze(0).cpu()
    split_ratios_flat = split_ratios.reshape(-1)
    num_hops_per_path = sample_data['paths_to_edges'].sum(dim=1)
    total_weighted_hops = (num_hops_per_path * split_ratios_flat).sum()
    return total_weighted_hops.item()

@app.route('/update_policy', methods=['POST'])
def update_policy():
    """API endpoint to update policy embeddings"""
    global policy_embeds, results
    
    try:
        data = request.get_json()
        
        if 'policy_embeds' not in data:
            return jsonify({'success': False, 'error': 'Missing policy_embeds parameter'}), 400
        
        new_policy_embeds = data['policy_embeds']
        
        # Validate policy_embeds
        if not isinstance(new_policy_embeds, list) or len(new_policy_embeds) != 3:
            return jsonify({'success': False, 'error': 'policy_embeds must be a list of 3 numbers'}), 400
        
        # Check if all values are numbers
        try:
            new_policy_embeds = [float(x) for x in new_policy_embeds]
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': 'All policy_embeds values must be numbers'}), 400
        
        # Update global policy_embeds
        policy_embeds = new_policy_embeds
        
        # Clear cached results to force recomputation with new policy embeddings
        results.clear()
        
        return jsonify({
            'success': True, 
            'policy_embeds': policy_embeds,
            'message': 'Policy embeddings updated successfully. Cache cleared.'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """API endpoint to clear cached results"""
    global results
    
    try:
        results.clear()
        return jsonify({
            'success': True,
            'message': f'Cache cleared. All {num_samples} samples will be recomputed on demand.'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get_policy', methods=['GET'])
def get_policy():
    """API endpoint to get current policy embeddings"""
    global policy_embeds
    
    return jsonify({
        'success': True,
        'policy_embeds': policy_embeds
    })

@app.route('/get_network_data')
def get_network_data():
    """Return the base network structure"""
    G = test_dataset.snapshot.graph
    
    # Scale coordinates
    min_lat = min(loc["latitude"] for loc in node_locations)
    max_lat = max(loc["latitude"] for loc in node_locations)
    min_lon = min(loc["longitude"] for loc in node_locations)
    max_lon = max(loc["longitude"] for loc in node_locations)
    
    # Read and encode router SVG icon
    router_icon_path = "assets/router.svg"
    with open(router_icon_path, 'rb') as f:
        svg_data = f.read()
    router_icon_base64 = base64.b64encode(svg_data).decode('utf-8')
    router_icon_uri = f"data:image/svg+xml;base64,{router_icon_base64}"
    
    # Prepare nodes data
    nodes_data = []
    for i, node in enumerate(G.nodes()):
        loc = node_locations[node]
        x = (loc["longitude"] - min_lon) / (max_lon - min_lon) * 1000 - 500
        y = -((loc["latitude"] - min_lat) / (max_lat - min_lat) * 500 - 250)
        
        nodes_data.append({
            'id': node,
            'x': x,
            'y': y,
            'title': f"Node {node}: {loc['city']}",
            'shape': 'image',
            'image': router_icon_uri,
            'size': 25
        })
    
    # Prepare edges data
    edges_data = []
    for source, target, data in G.edges(data=True):
        capacity = float(data['capacity'])
        width = 1 + capacity / 3
        
        edges_data.append({
            'id': f"{source}-{target}",
            'from': source,
            'to': target,
            'width': 2*width,
            'title': f"Capacity: {capacity} Gbps",
            'color': "#808080"
        })
    
    # Network options
    options = {
        "physics": {
            "enabled": False,
            "stabilization": False
        },
        "interaction": {
            "hover": True,
            "navigationButtons": False,
            "multiselect": True,
            "dragNodes": True
        },
        "edges": {
            "smooth": False,
            "arrows": {
                "to": {"enabled": False},
                "from": {"enabled": False}
            }
        }
    }
    
    return jsonify({
        'nodes': nodes_data,
        'edges': edges_data,
        'options': options
    })

@app.route('/get_sample_data')
def get_sample_data():
    """Return only the updated data for a specific sample"""
    sample_idx = request.args.get('sample', default=0, type=int)
    
    if sample_idx >= num_samples:
        sample_idx = 0
    
    # Always run inference for this sample
    out = pipe(test_dataset[sample_idx], policy_embeds=policy_embeds, num_for_loops=num_for_loops, 
                num_paths_per_pair=num_paths_per_pair, return_loss=True)
    results[sample_idx] = out
    out_queue.append(out)
    
    # Get edge colors for this sample
    edge_colors = get_edge_colors(sample_idx, out)
    cost = calculate_cost(out, test_dataset[sample_idx])
    
    return jsonify({
        'edge_colors': edge_colors,
        'norm_mlu': out.loss.norm_mlu.item(),
        'max_sensitivity': out.loss.max_sensitivity.item(),
        'cost': cost
    })

def get_edge_colors(sample_idx, out):
    """Calculate edge colors based on utilization"""
    G = test_dataset.snapshot.graph
    
    # Get edge utilization data
    edges_util = out.edges_util.cpu().numpy()/1000  # convert traffic to Gbps
    edge_index = test_dataset[sample_idx]['edge_index']
    edges_util = edges_util * 300  # expand 200 times for better visualization
    
    # Create color map
    cmap = plt.cm.RdYlGn_r  # Red-Yellow-Green colormap, reversed
    
    # Create an edge utilization dictionary
    edge_util_dict = {}
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        util = float(edges_util[0, i])
        edge_util_dict[(src, dst)] = util
    
    # Calculate colors for each edge
    edge_colors = {}
    for source, target, data in G.edges(data=True):
        capacity = float(data['capacity'])
        
        # Get utilization for this edge
        util_forward = edge_util_dict.get((source, target), 0)
        util_backward = edge_util_dict.get((target, source), 0)
        util = max(util_forward, util_backward)
        
        # Cap at 100% for coloring
        util_capped = min(util, 1.0)
        
        # Map utilization to color
        color_rgb = cmap(util_capped)
        color_hex = '#%02x%02x%02x' % (int(color_rgb[0]*255), int(color_rgb[1]*255), int(color_rgb[2]*255))
        
        edge_colors[f"{source}-{target}"] = {
            'color': color_hex,
            'title': f"Capacity: {capacity} Gbps\nUtilization: {util*100:.1f}%"
        }
    
    return edge_colors

@app.route('/api/policy', methods=['GET', 'POST', 'PUT'])
def api_policy():
    """RESTful API endpoint for policy embeddings management"""
    global policy_embeds, results
    
    if request.method == 'GET':
        # Get current policy embeddings
        return jsonify({
            'success': True,
            'policy_embeds': policy_embeds,
            'cache_size': len(results)
        })
    
    elif request.method in ['POST', 'PUT']:
        # Update policy embeddings
        try:
            data = request.get_json()
            
            if 'policy_embeds' not in data:
                return jsonify({'success': False, 'error': 'Missing policy_embeds parameter'}), 400
            
            new_policy_embeds = data['policy_embeds']
            
            # Validate policy_embeds
            if not isinstance(new_policy_embeds, list) or len(new_policy_embeds) != 3:
                return jsonify({'success': False, 'error': 'policy_embeds must be a list of 3 numbers'}), 400

            # Check if all values are numbers
            try:
                new_policy_embeds = [float(x) for x in new_policy_embeds]
            except (ValueError, TypeError):
                return jsonify({'success': False, 'error': 'All policy_embeds values must be numbers'}), 400
            
            # Optional: validate range if needed
            if 'validate_range' in data and data['validate_range']:
                for i, val in enumerate(new_policy_embeds):
                    if val < 0 or val > 1:  # Example range
                        return jsonify({
                            'success': False, 
                            'error': f'Policy embedding at index {i} ({val}) is out of range [-10, 10]'
                        }), 400
            
            # Store old values for comparison
            old_policy_embeds = policy_embeds.copy()
            
            # Update global policy_embeds
            policy_embeds = new_policy_embeds
            
            # Clear cache if requested or by default
            clear_cache = data.get('clear_cache', True)
            if clear_cache:
                results.clear()
                cache_message = 'Cache cleared.'
            else:
                cache_message = 'Cache retained.'
            
            return jsonify({
                'success': True,
                'old_policy_embeds': old_policy_embeds,
                'new_policy_embeds': policy_embeds,
                'message': f'Policy embeddings updated successfully. {cache_message}',
                'cache_cleared': clear_cache
            })
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
            
@app.route('/api/topology', methods=['GET'])
def get_topology():
    """Return concise network topology information"""
    G = test_dataset.snapshot.graph
    
    # Get nodes with simplified location data
    nodes = []
    for node_id in G.nodes():
        loc = next((l for l in node_locations if l["name"] == f"{G.nodes[node_id].get('name', node_id)}"), None)
        nodes.append({
            'id': node_id,
            'name': G.nodes[node_id].get('name', f"Node {node_id}"),
            'city': loc['city'] if loc else None,
            'lat': loc['latitude'] if loc else None,
            'lon': loc['longitude'] if loc else None
        })
    
    # Get edges with capacity information
    edges = []
    for source, target, data in G.edges(data=True):
        edges.append({
            'source': source,
            'target': target,
            'id': f"{source}-{target}",
            'capacity': float(data['capacity']) 
        })
    
    return jsonify({
        'nodes': nodes,
        'edges': edges,
        'node_count': len(nodes),
        'edge_count': len(edges)
    })

@app.route('/api/topology/stats', methods=['GET'])
def get_topology_statistics():
    """Return comprehensive statistical analysis of the network topology with rounded values for LLM agents"""
    G = test_dataset.snapshot.graph
    is_directed = nx.is_directed(G)
    
    # Basic network statistics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = round(nx.density(G), 3)
    
    # Compute degree statistics
    if is_directed:
        in_degrees = [d for _, d in G.in_degree()]
        out_degrees = [d for _, d in G.out_degree()]
        total_degrees = [d for _, d in G.degree()]
        
        degree_stats = {
            'in_degree': {
                'min': min(in_degrees) if in_degrees else 0,
                'max': max(in_degrees) if in_degrees else 0,
                'mean': round(sum(in_degrees) / len(in_degrees), 2) if in_degrees else 0,
                'median': round(float(np.median(in_degrees)), 2) if in_degrees else 0
            },
            'out_degree': {
                'min': min(out_degrees) if out_degrees else 0,
                'max': max(out_degrees) if out_degrees else 0,
                'mean': round(sum(out_degrees) / len(out_degrees), 2) if out_degrees else 0,
                'median': round(float(np.median(out_degrees)), 2) if out_degrees else 0
            },
            'total_degree': {
                'min': min(total_degrees) if total_degrees else 0,
                'max': max(total_degrees) if total_degrees else 0,
                'mean': round(sum(total_degrees) / len(total_degrees), 2) if total_degrees else 0,
                'median': round(float(np.median(total_degrees)), 2) if total_degrees else 0
            }
        }
    else:
        degrees = [d for _, d in G.degree()]
        degree_stats = {
            'min': min(degrees) if degrees else 0,
            'max': max(degrees) if degrees else 0,
            'mean': round(sum(degrees) / len(degrees), 2) if degrees else 0,
            'median': round(float(np.median(degrees)), 2) if degrees else 0
        }
    
    # Connected components analysis
    component_stats = {}
    
    if is_directed:
        strongly_connected = list(nx.strongly_connected_components(G))
        weakly_connected = list(nx.weakly_connected_components(G))
        
        component_stats = {
            'is_strongly_connected': nx.is_strongly_connected(G),
            'is_weakly_connected': nx.is_weakly_connected(G),
            'num_strongly_connected': len(strongly_connected),
            'num_weakly_connected': len(weakly_connected),
            'largest_strongly_component_size': len(max(strongly_connected, key=len)) if strongly_connected else 0,
            'largest_weakly_component_size': len(max(weakly_connected, key=len)) if weakly_connected else 0
        }
        
        # Compute diameter on largest weakly connected component
        largest_wcc = max(weakly_connected, key=len) if weakly_connected else []
        if largest_wcc:
            try:
                wcc_subgraph = G.subgraph(largest_wcc).copy()
                if nx.is_strongly_connected(wcc_subgraph):
                    component_stats['largest_wcc_diameter'] = nx.diameter(wcc_subgraph)
                else:
                    component_stats['largest_wcc_diameter'] = 'Not strongly connected'
            except Exception:
                pass
    else:
        connected_components = list(nx.connected_components(G))
        
        component_stats = {
            'is_connected': nx.is_connected(G),
            'num_connected_components': len(connected_components),
            'largest_component_size': len(max(connected_components, key=len)) if connected_components else 0
        }
        
        if nx.is_connected(G):
            try:
                component_stats['diameter'] = nx.diameter(G)
                component_stats['avg_shortest_path'] = round(nx.average_shortest_path_length(G), 2)
            except Exception:
                pass
        elif connected_components:
            largest_cc = max(connected_components, key=len)
            try:
                cc_subgraph = G.subgraph(largest_cc)
                component_stats['largest_cc_diameter'] = nx.diameter(cc_subgraph)
                component_stats['largest_cc_avg_path'] = round(nx.average_shortest_path_length(cc_subgraph), 2)
            except Exception:
                pass
    
    # Clustering coefficient
    try:
        clustering_coefficient = round(nx.average_clustering(G), 3)
        clustering_stats = {'avg_clustering_coefficient': clustering_coefficient}
    except Exception:
        clustering_stats = {'clustering_error': 'Could not compute clustering coefficient'}
    
    # Capacity statistics
    capacity_stats = None
    if any('capacity' in G.edges[e] for e in G.edges):
        try:
            capacities = [float(G.edges[e].get('capacity', 0)) for e in G.edges]
            capacity_stats = {
                'min': round(min(capacities), 2) if capacities else 0,
                'max': round(max(capacities), 2) if capacities else 0,
                'mean': round(sum(capacities) / len(capacities), 2) if capacities else 0,
                'total': round(sum(capacities), 2) if capacities else 0,
                'median': round(float(np.median(capacities)), 2) if capacities else 0,
                'units': 'Gbps' 

            }
        except Exception:
            capacity_stats = {'capacity_error': 'Could not compute capacity statistics'}
    
    # Centrality statistics for smaller networks
    centrality_stats = {}
    if num_nodes <= 1000:
        try:
            sample_size = min(num_nodes, 100)
            betweenness = nx.betweenness_centrality(G, k=sample_size)
            betw_values = list(betweenness.values())
            
            centrality_stats = {
                'betweenness': {
                    'max_node': max(betweenness.items(), key=lambda x: x[1])[0],
                    'max_value': round(max(betw_values), 3) if betw_values else 0,
                    'min_value': round(min(betw_values), 3) if betw_values else 0,
                    'mean': round(sum(betw_values) / len(betw_values), 3) if betw_values else 0,
                    'sample_size': sample_size
                }
            }
        except Exception:
            centrality_stats = {'centrality_error': 'Could not compute centrality metrics'}

    result = {
        'basic_stats': {
            'node_count': num_nodes,
            'edge_count': num_edges,
            'density': density,
            'is_directed': is_directed
        },
        'degree_stats': degree_stats,
        'component_stats': component_stats,
        'clustering_stats': clustering_stats,
        'capacity_stats': capacity_stats,
        'centrality_stats': centrality_stats
    }
    
    
    # Final formatting pass to ensure all floats are rounded
    result = round_floats_recursive(result)
    
    return jsonify(result)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get current network statistics with optional delay"""
    global policy_embeds, out_queue
    
    # Get wait time parameter from query string, default to 0
    wait_time = request.args.get('wait', default=0, type=float)
    
    # Wait for the specified time if greater than 0
    if wait_time > 0:
        time.sleep(wait_time)
    
    if len(out_queue) > 0:
        mlu_list = [out.loss.mlu.cpu().numpy() for out in out_queue]
        max_sensitivity = [out.loss.max_sensitivity.cpu().numpy() for out in out_queue]
        norm_mlu_list = [out.loss.norm_mlu.item() for out in out_queue]
        
        # Initialize cost list
        all_cost = []
        
        # Process each output to calculate cost
        for i, out in enumerate(out_queue):
            split_ratios = out.split_ratios.squeeze(0).cpu()
            split_ratios_flat = split_ratios.reshape(-1)
            num_hops_per_path = test_dataset[i]['paths_to_edges'].sum(dim=1)
            total_weighted_hops = (num_hops_per_path * split_ratios_flat).sum()
            all_cost.append(total_weighted_hops.item())  # Convert to Python float
        stats = {
            'mlu': {
                'min': float(np.min(mlu_list)),
                'max': float(np.max(mlu_list)),
                'mean': float(np.mean(mlu_list)),
                'median': float(np.median(mlu_list)),
                'std': float(np.std(mlu_list))
            },
            'max_sensitivity': {
                'min': float(np.min(max_sensitivity)),
                'max': float(np.max(max_sensitivity)),
                'mean': float(np.mean(max_sensitivity)),
                'median': float(np.median(max_sensitivity)),
                'std': float(np.std(max_sensitivity))
            },
            'cost': {
                'min': float(np.min(all_cost)),
                'max': float(np.max(all_cost)),
                'mean': float(np.mean(all_cost)),
                'median': float(np.median(all_cost)),
                'std': float(np.std(all_cost))
            },
            'policy_embeddings': policy_embeds,
            'samples_processed': len(out_queue),
            'wait_time_applied': wait_time
        }
        rounded_stats = round_floats_recursive(stats, 3)
        
        return jsonify(rounded_stats)
    else:
        return jsonify({
            'error': 'No data available',
            'policy_embeddings': policy_embeds,
            'samples_processed': 0,
            'wait_time_applied': wait_time
        })


if __name__ == '__main__':
    # Create static directory if it doesn't exist
    os.makedirs('assets', exist_ok=True)
    
    print(f"""
    ========================================
    Network Topology Visualization Server
    ========================================
    Total Samples: {num_samples}
    Initial Policy Embeddings: {policy_embeds}
    GPU Device: cuda:{gpu_id}
    
    Starting server on http://0.0.0.0:5000
    ========================================
    """)
    
    app.run(host='0.0.0.0', port=5000, debug=True)

    
