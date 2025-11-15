import { useState, useEffect, useMemo } from 'react'
import './Home.css'
import HeroSection from '../components/HeroSection'
import SearchBar from '../components/SearchBar'
import FilterSection from '../components/FilterSection'
import InventoryGrid from '../components/InventoryGrid'

function Home() {
  const [inventory, setInventory] = useState([])
  const [loading, setLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedStockStatus, setSelectedStockStatus] = useState('All')
  const [sortBy, setSortBy] = useState('name')
  const stockStatusOptions = ['All', 'In Stock', 'Low Stock', 'Out of Stock']

  const statusLabels = {
    'in-stock': 'In Stock',
    'low-stock': 'Low Stock',
    'out-of-stock': 'Out of Stock'
  }

  const getStockStatus = (item) => {
    if (item.quantity === 0) return 'out-of-stock'
    if (item.quantity <= item.threshold) return 'low-stock'
    return 'in-stock'
  }

  // Fetch real inventory data from backend API
  useEffect(() => {
    fetch("http://127.0.0.1:8080/api/inventory") // Replace with your API port
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          const formatted = data.items.map(item => ({
            id: item.inventory_id,
            name: item.item_name,
            category: item.item_type,
            quantity: item.initial_stock || 0,
            threshold: item.minimum_required || 0
          }))
          setInventory(formatted)
        }
        setLoading(false)
      })
      .catch(err => {
        console.error("Error fetching inventory:", err)
        setLoading(false)
      })
  }, [])

  const filteredAndSortedInventory = useMemo(() => {
    let filtered = inventory.filter(item => {
      const matchesSearch = item.name.toLowerCase().includes(searchTerm.toLowerCase())
      const status = getStockStatus(item)
      const statusLabel = statusLabels[status]
      const matchesStockStatus = selectedStockStatus === 'All' || statusLabel === selectedStockStatus
      return matchesSearch && matchesStockStatus
    })

    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'name':
          return a.name.localeCompare(b.name)
        case 'quantity':
          return b.quantity - a.quantity
        case 'category':
          return a.category.localeCompare(b.category)
        case 'status':
          const statusOrder = { 'out-of-stock': 0, 'low-stock': 1, 'in-stock': 2 }
          return statusOrder[getStockStatus(a)] - statusOrder[getStockStatus(b)]
        default:
          return 0
      }
    })

    return filtered
  }, [inventory, searchTerm, selectedStockStatus, sortBy])

  const statusColors = {
    'in-stock': '#10b981',
    'low-stock': '#f59e0b',
    'out-of-stock': '#ef4444'
  }

  // Calculate stats
  const stats = {
    total: inventory.length,
    inStock: inventory.filter(item => getStockStatus(item) === 'in-stock').length,
    lowStock: inventory.filter(item => getStockStatus(item) === 'low-stock').length,
    outOfStock: inventory.filter(item => getStockStatus(item) === 'out-of-stock').length
  }

  if (loading) {
    return <div className="loading">Loading inventory...</div>
  }

  return (
    <div className="home-page">
      <HeroSection stats={stats} />

      <div className="search-filter-section">
        <SearchBar searchTerm={searchTerm} setSearchTerm={setSearchTerm} />
        <FilterSection
          sortBy={sortBy}
          setSortBy={setSortBy}
          selectedStockStatus={selectedStockStatus}
          setSelectedStockStatus={setSelectedStockStatus}
          stockStatusOptions={stockStatusOptions}
        />
      </div>

      <InventoryGrid
        filteredAndSortedInventory={filteredAndSortedInventory}
        getStockStatus={getStockStatus}
        statusLabels={statusLabels}
        statusColors={statusColors}
      />
    </div>
  )
}

export default Home
