import { useState, useEffect } from 'react'
import { useTheme } from '../contexts/ThemeContext'
import StatsGrid from '../components/StatsGrid'
import ChartsSection from '../components/ChartsSection'
import LowStockAlerts from '../components/LowStockAlerts'
import AddItemForm from '../components/AddItemForm'
import InventoryTable from '../components/InventoryTable'
import './Dashboard.css'

function Dashboard() {
  const { theme } = useTheme()
  const [inventory, setInventory] = useState([])
  const [loading, setLoading] = useState(true)
  const [selectedItemId, setSelectedItemId] = useState('')
  const [editingId, setEditingId] = useState(null)
  const [editForm, setEditForm] = useState({ name: '', category: '', quantity: '', threshold: '' })
  const [newItem, setNewItem] = useState({ name: '', category: '', quantity: '', threshold: '' })
  const [showAddForm, setShowAddForm] = useState(false)

  // Fetch real inventory data
  useEffect(() => {
    fetch("http://127.0.0.1:8080/api/inventory")
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          const formatted = data.items.map(item => ({
            id: item.inventory_id,
            name: item.item_name,
            category: item.item_type || 'Unknown',
            quantity: item.initial_stock || 0,
            threshold: item.minimum_required || 0
          }))
          setInventory(formatted)
        }
        setLoading(false)
      })
      .catch(err => {
        console.error(err)
        setLoading(false)
      })
  }, [])

  if (loading) {
    return <div className="loading">Loading inventory...</div>
  }

  const chartColors = {
    dark: { bg: '#1e293b', border: '#334155', text: '#f1f5f9', textSecondary: '#cbd5e1', grid: '#334155' },
    light: { bg: '#ffffff', border: '#e2e8f0', text: '#1e293b', textSecondary: '#64748b', grid: '#e2e8f0' }
  }
  const colors = chartColors[theme]

  const getStockStatus = (item) => {
    if (item.quantity === 0) return 'out-of-stock'
    if (item.quantity <= item.threshold) return 'low-stock'
    return 'in-stock'
  }

  const stats = {
    totalItems: inventory.length,
    inStock: inventory.filter(item => item.quantity > item.threshold).length,
    lowStock: inventory.filter(item => item.quantity > 0 && item.quantity <= item.threshold).length,
    outOfStock: inventory.filter(item => item.quantity === 0).length,
    totalQuantity: inventory.reduce((sum, item) => sum + item.quantity, 0)
  }

  const categoryData = inventory.reduce((acc, item) => {
    if (!acc[item.category]) acc[item.category] = { name: item.category, quantity: 0, items: 0 }
    acc[item.category].quantity += item.quantity
    acc[item.category].items += 1
    return acc
  }, {})

  const categoryChartData = Object.values(categoryData).map(cat => ({
    name: cat.name,
    quantity: cat.quantity,
    items: cat.items
  }))

  const statusData = [
    { name: 'In Stock', value: stats.inStock, color: '#10b981' },
    { name: 'Low Stock', value: stats.lowStock, color: '#f59e0b' },
    { name: 'Out of Stock', value: stats.outOfStock, color: '#ef4444' }
  ]

  const consumptionData = [
    { month: 'Jan', usage: 450 },
    { month: 'Feb', usage: 520 },
    { month: 'Mar', usage: 480 },
    { month: 'Apr', usage: 610 },
    { month: 'May', usage: 550 },
    { month: 'Jun', usage: 680 }
  ]

  const lowStockAlerts = inventory.filter(item => item.quantity <= item.threshold)

  const handleEdit = (item) => {
    setEditingId(item.id)
    setEditForm({
      name: item.name,
      category: item.category,
      quantity: item.quantity.toString(),
      threshold: item.threshold.toString()
    })
  }

  const handleSaveEdit = () => {
    setInventory(inventory.map(item =>
      item.id === editingId
        ? { ...item, name: editForm.name, category: editForm.category, quantity: parseInt(editForm.quantity) || 0, threshold: parseInt(editForm.threshold) || 0 }
        : item
    ))
    setEditingId(null)
    setEditForm({ name: '', category: '', quantity: '', threshold: '' })
  }

  const handleCancelEdit = () => {
    setEditingId(null)
    setEditForm({ name: '', category: '', quantity: '', threshold: '' })
  }

  const categories = [...new Set(inventory.map(item => item.category))]

  return (
    <div className="dashboard-page">
      <div className="page-header" style={{ marginBottom: '1rem' }}>
        <h1 className="page-title">Dashboard</h1>
        {/* Dropdown to select inventory item */}
        <select
          value={selectedItemId}
          onChange={(e) => setSelectedItemId(e.target.value)}
          style={{ marginTop: '1rem', padding: '0.5rem' }}
        >
          <option value="">-- Select Inventory Item --</option>
          {inventory.map(item => (
            <option key={item.id} value={item.id}>{item.name}</option>
          ))}
        </select>
      </div>

      <StatsGrid stats={stats} />

      <ChartsSection
        categoryChartData={categoryChartData}
        statusData={statusData}
        consumptionData={consumptionData}
        colors={colors}
      />

      <LowStockAlerts lowStockAlerts={lowStockAlerts} />

      <div className="management-section">
        <div className="section-header">
          <h2 className="section-title">Inventory Management</h2>
          <button onClick={() => setShowAddForm(!showAddForm)} className="add-button">
            {showAddForm ? 'Cancel' : '+ Add New Item'}
          </button>
        </div>

        {showAddForm && (
          <AddItemForm
            newItem={newItem}
            setNewItem={setNewItem}
            categories={categories}
            handleAddItem={() => {}}
          />
        )}

        <InventoryTable
          inventory={inventory}
          editingId={editingId}
          editForm={editForm}
          setEditForm={setEditForm}
          categories={categories}
          getStockStatus={getStockStatus}
          handleEdit={handleEdit}
          handleSaveEdit={handleSaveEdit}
          handleCancelEdit={handleCancelEdit}
          handleDelete={() => {}}
        />
      </div>
    </div>
  )
}

export default Dashboard
