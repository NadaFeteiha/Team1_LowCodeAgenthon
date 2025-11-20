import { Bar, Pie, Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from "chart.js";
import "./ItemForecastModal.css";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

export default function ItemForecastModal({ item, parsed, onClose }) {
  if (!item || !parsed) return null;

  const { currentStock, reorderLevel, lowStock, actions } = parsed;

  // ---------------------------
  // Bar Chart: Stock vs Reorder
  // ---------------------------
  const stockBarData = {
    labels: ["Stock", "Reorder Level"],
    datasets: [
      {
        label: "Units",
        data: [currentStock, reorderLevel],
        backgroundColor: ["rgba(16,185,129,0.6)", "rgba(239,68,68,0.6)"],
        borderColor: ["green", "red"],
        borderWidth: 2
      }
    ]
  };

  // ---------------------------
  // Pie Chart: Threshold Status
  // ---------------------------
  const thresholdPieData = {
    labels: ["Stock OK", "Low Stock"],
    datasets: [
      {
        data: lowStock ? [0, 1] : [1, 0],
        backgroundColor: ["rgba(16,185,129,0.6)", "rgba(239,68,68,0.6)"]
      }
    ]
  };

  // ---------------------------
  // Line Chart: Stock Trend (Example Data)
  // ---------------------------
  const stockTrendData = {
    labels: ["Week 1", "Week 2", "Week 3", "Week 4"],
    datasets: [
      {
        label: "Stock Level",
        data: [currentStock, currentStock - 2, currentStock - 5, reorderLevel],
        borderColor: "rgba(16,185,129,0.8)",
        backgroundColor: "rgba(16,185,129,0.3)",
        fill: true,
        tension: 0.4,
        pointRadius: 5,
        pointHoverRadius: 7
      }
    ]
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <h2>{item.name} — Inventory Summary</h2>
        <button className="close-btn" onClick={onClose}>
          X
        </button>

        {/* Red Alert Banner */}
        {lowStock && (
          <div className="low-stock-alert">
            ⚠️ {currentStock === 0 ? "Item Out of Stock!" : "Low Stock Alert!"}
          </div>
        )}

        {/* Stock Bar */}
        <div className="chart-container">
          <h4>Stock vs Threshold</h4>
          <Bar data={stockBarData} />
        </div>

        {/* Pie */}
        <div className="chart-container">
          <h4>Threshold Status</h4>
          <Pie data={thresholdPieData} />
        </div>

        {/* Line */}
        <div className="chart-container">
          <h4>Stock Trend (Last 4 Weeks)</h4>
          <Line data={stockTrendData} />
        </div>

        {/* Actions */}
        <div className="actions-box">
          <h4>Recommended Actions</h4>
          <ul>
            {actions.map((a, idx) => (
              <li key={idx}>{a}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
