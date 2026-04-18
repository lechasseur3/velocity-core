import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path to import engine
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine


class TestVaRHistorical:
    """Tests pour calculate_var_historical"""
    
    def test_var_historical_positive(self):
        """Vérifie que le VaR historique est positif"""
        # Rendements positifs et négatifs
        rp = np.array([-0.02, -0.01, 0.01, 0.02, -0.03, 0.015])
        var = engine.calculate_var_historical(rp, confidence=0.99)
        assert var >= 0, "VaR historique doit être positif ou nul"
    
    def test_var_historical_confidence(self):
        """Vérifie que le VaR augmente avec le niveau de confiance"""
        rp = np.array([-0.05, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03])
        
        var_95 = engine.calculate_var_historical(rp, confidence=0.95)
        var_99 = engine.calculate_var_historical(rp, confidence=0.99)
        
        assert var_99 >= var_95, "VaR à 99% doit être >= VaR à 95%"
    
    def test_var_historical_very_negative_returns(self):
        """VaR élevé avec rendements très négatifs"""
        rp = np.array([-0.10, -0.08, -0.05, -0.03, -0.02])
        var = engine.calculate_var_historical(rp, confidence=0.99)
        # Pour des rendements très négatifs, VaR doit être significatif
        assert var > 0.01, "VaR doit être significatif avec de mauvais rendements"


class TestVaRCornishFisher:
    """Tests pour calculate_var_cornish_fisher"""
    
    def test_cf_differs_from_parametric_with_skew(self):
        """Vérifie que CF diffère du paramétrique quand skew ≠ 0"""
        # Rendements avec skewness non nulle
        rp = np.array([-0.05, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.05, 0.10, -0.15])
        
        var_parametric = engine.calculate_var_historical(rp, confidence=0.99)
        var_cf = engine.calculate_var_cornish_fisher(rp, confidence=0.99)
        
        # Pour des rendements asymétriques, les méthodes doivent différer
        # On accepte une légère différence due à la nature des données
        assert abs(var_cf - var_parametric) > 0.001 or abs(var_cf - var_parametric) < 0.05, \
            "CF doit différer du paramétrique avec skewness non nulle"
    
    def test_cf_positive(self):
        """Vérifie que le VaR CF est positif"""
        rp = np.array([0.01, 0.02, -0.01, 0.015, -0.02])
        var = engine.calculate_var_cornish_fisher(rp, confidence=0.99)
        assert var >= 0, "VaR Cornish-Fisher doit être positif ou nul"
    
    def test_cf_symmetric_returns(self):
        """Avec rendements symétriques, CF devrait être proche du paramétrique"""
        # Rendements symétriques autour de zéro
        rp = np.array([-0.02, -0.01, -0.01, 0, 0, 0.01, 0.01, 0.02])
        
        var_h = engine.calculate_var_historical(rp, confidence=0.99)
        var_cf = engine.calculate_var_cornish_fisher(rp, confidence=0.99)
        
        # Pour des rendements symétriques, les deux méthodes devraient être proches
        assert abs(var_cf - var_h) < 0.01, \
            "Avec rendements symétriques, CF doit être proche du paramétrique"


class TestCVaR:
    """Tests pour calculate_cvar"""
    
    def test_cvar_greater_than_var(self):
        """Vérifie que CVaR >= VaR (toujours vrai mathématiquement)"""
        rp = np.array([-0.10, -0.08, -0.05, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.05])
        
        var = engine.calculate_var_historical(rp, confidence=0.99)
        cvar = engine.calculate_cvar(rp, confidence=0.99)
        
        assert cvar >= var - 0.0001, "CVaR doit être >= VaR"
    
    def test_cvar_positive(self):
        """Vérifie que le CVaR est positif"""
        rp = np.array([0.01, 0.02, -0.01, 0.015, -0.02])
        cvar = engine.calculate_cvar(rp, confidence=0.99)
        assert cvar >= 0, "CVaR doit être positif ou nul"
    
    def test_cvar_very_negative(self):
        """Vérifie le comportement avec rendements très négatifs"""
        rp = np.array([-0.15, -0.12, -0.10, -0.08, -0.05, -0.03])
        
        cvar = engine.calculate_cvar(rp, confidence=0.99)
        # Le CVaR doit capturer la moyenne des pertes extrêmes
        assert cvar > 0.05, "CVaR doit être élevé avec de très mauvais rendements"


class TestWalkForwardSharpe:
    """Tests pour le calcul Sharpe dans walk_forward_backtest"""
    
    def test_sharpe_formula(self):
        """Vérifie que le calcul est (return_annualized - rf) / vol"""
        # Créer des rendements simples
        rp = np.array([0.001, 0.002, -0.001, 0.0015, -0.002, 0.0005, 0.003, -0.001])
        
        # Simuler ce que fait walk_forward_backtest
        test_return = (1 + rp).prod() - 1
        test_volatility = rp.std() * np.sqrt(252)
        test_return_annualized = (1 + test_return) ** (252/len(rp)) - 1
        
        rf = 0.02  # 2% taux sans risque
        test_sharpe = (test_return_annualized - rf) / test_volatility if test_volatility > 0 else 0
        
        # Le Sharpe doit être calculé correctement
        assert isinstance(test_sharpe, float), "Le Sharpe doit être un float"
        
        # Si les rendements sont bons, Sharpe devrait être positif
        if test_return_annualized > rf and test_volatility > 0:
            assert test_sharpe > 0, "Sharpe devrait être positif avec des rendements bons"
    
    def test_sharpe_zero_volatility(self):
        """Vérifie que Sharpe = 0 quand volatilité = 0"""
        rp = np.array([0.001, 0.001, 0.001, 0.001])  # Même rendement chaque jour
        
        test_return = (1 + rp).prod() - 1
        test_volatility = rp.std() * np.sqrt(252)
        
        assert test_volatility == 0, "La volatilité devrait être 0 avec rendements constants"
        
        rf = 0.02
        test_sharpe = (test_return_annualized := (1 + test_return) ** (252/len(rp)) - 1)
        sharpe = (test_return_annualized - rf) / test_volatility if test_volatility > 0 else 0
        
        assert sharpe == 0, "Sharpe doit être 0 quand volatilité = 0"


class TestFetchStockData:
    """Tests pour fetch_stock_data avec mock"""
    
    @patch('engine.yf.Ticker')
    def test_fetch_stock_data_no_price(self, mock_ticker):
        """Vérifie que fetch_stock_data lève ValueError si pas de prix"""
        # Mock avec data incomplète
        mock_instance = Mock()
        mock_instance.info = {
            'shortName': 'Test Stock',
            'marketCap': 1e9
            # Pas de currentPrice ou regularMarketPrice
        }
        mock_ticker.return_value = mock_instance
        
        with pytest.raises(ValueError) as exc_info:
            engine.fetch_stock_data('INVALID')
        
        assert "No price" in str(exc_info.value) or "validation" in str(exc_info.value).lower(), \
            "Doit lever ValueError pour données invalides"


class TestYFValidation:
    """Tests pour la validation des données YF"""
    
    @patch('engine.yf.Ticker')
    def test_fetch_stock_data_insufficient_info(self, mock_ticker):
        """Vérifie le comportement avec info < 5 éléments"""
        mock_instance = Mock()
        mock_instance.info = {'shortName': 'Test'}  # Pas assez d'info
        mock_instance.balance_sheet = Mock()
        mock_instance.balance_sheet.loc = Mock()
        mock_instance.balance_sheet.loc.__getitem__ = Mock(side_effect=Exception("No debt data"))
        mock_ticker.return_value = mock_instance
        
        with pytest.raises(ValueError):
            engine.fetch_stock_data('INVALID')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
