'use client'

import { Heart, Menu, X } from 'lucide-react'
import { useState } from 'react'

export default function Header() {
  const [isMenuOpen, setIsMenuOpen] = useState(false)

  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <div className="flex items-center">
            <div className="flex-shrink-0 flex items-center">
              <Heart className="w-8 h-8 text-primary-600 mr-2" />
              <span className="text-xl font-bold text-gray-900">CVML</span>
            </div>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex space-x-8">
            <a href="#" className="text-gray-900 hover:text-primary-600 px-3 py-2 text-sm font-medium">
              Home
            </a>
            <a href="#" className="text-gray-500 hover:text-primary-600 px-3 py-2 text-sm font-medium">
              About
            </a>
            <a href="#" className="text-gray-500 hover:text-primary-600 px-3 py-2 text-sm font-medium">
              Features
            </a>
            <a href="#" className="text-gray-500 hover:text-primary-600 px-3 py-2 text-sm font-medium">
              Contact
            </a>
          </nav>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="text-gray-500 hover:text-gray-900 focus:outline-none focus:text-gray-900"
            >
              {isMenuOpen ? (
                <X className="w-6 h-6" />
              ) : (
                <Menu className="w-6 h-6" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {isMenuOpen && (
          <div className="md:hidden">
            <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3 border-t border-gray-200">
              <a href="#" className="text-gray-900 block px-3 py-2 text-base font-medium">
                Home
              </a>
              <a href="#" className="text-gray-500 hover:text-primary-600 block px-3 py-2 text-base font-medium">
                About
              </a>
              <a href="#" className="text-gray-500 hover:text-primary-600 block px-3 py-2 text-base font-medium">
                Features
              </a>
              <a href="#" className="text-gray-500 hover:text-primary-600 block px-3 py-2 text-base font-medium">
                Contact
              </a>
            </div>
          </div>
        )}
      </div>
    </header>
  )
}
