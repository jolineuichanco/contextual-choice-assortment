'''
wrapperClass.py

Classes for choice models and products used in assortment optimization.

This module implements:
- Product: A product with price, margin, and attributes
- MNLChoiceModel: Standard Multinomial Logit choice model
- ChoiceModel: Contextual Concavity (CC) choice model

The CC model captures reference-dependent choice behavior where consumers
evaluate products relative to the "best" option in each attribute dimension.
See Section 2 of the paper for model details.

Author: Joline Uichanco
Created: Jan 3, 2022
'''

import numpy as np

class Product():
    '''
    A product with attributes and margin.
    
    Attributes
    ----------
    product_id : str
        Unique identifier for the product (e.g., "P01", "P02")
    margin : float
        Profit margin (price - cost) for this product
    theta : list
        Attribute vector where:
        - theta[0] = price (p_j)
        - theta[k] for k >= 1 = non-price attribute level for attribute k
    '''
    def __init__(self, product_id, margin, theta):
        '''
        Initialize a Product.
        
        Parameters
        ----------
        product_id : str
            Unique identifier for the product
        margin : float
            Profit margin (price - cost)
        theta : list
            Attribute vector [price, attr_1, attr_2, ..., attr_K]     
        '''
        self.product_id = product_id
        self.margin = margin
        self.theta = theta


class MNLChoiceModel():
    '''
    Multinomial Logit (MNL) Choice Model.
    
    The MNL model assumes choice probabilities of the form:
    
        P_j(S) = exp(V_j) / (v_0 + sum_{i in S} exp(V_i))
    
    where the utility V_j is linear in attributes:
    
        V_j = beta_0 - beta_p * p_j + sum_k beta_k * theta_{jk}
    
    This is a "misspecified" model that ignores contextual effects.
    See Section 6.2 of the paper.
    
    Attributes
    ----------
    beta_0 : float
        Intercept term (estimated)
    beta_price : float
        Price sensitivity coefficient (estimated, typically negative)
    beta_theta : list
        Coefficients for non-price attributes [beta_1, ..., beta_K]
    v0 : float
        No-purchase utility weight (normalized to 1)
    num_attributes : int
        Number of non-price attributes (K)
    '''
    
    def __init__(self, beta_0, beta_price, beta_theta, nochoice_weight):
        '''
        Initialize an MNL choice model with estimated parameters.
        
        Parameters
        ----------
        beta_0 : float
            Intercept term
        beta_price : float
            Price sensitivity coefficient
        beta_theta : list
            List of coefficients for non-price attributes
        nochoice_weight : float
            No-purchase utility weight (v_0)
        '''
        self.num_attributes = len(beta_theta)
        self.beta_price = beta_price
        self.beta_0 = beta_0
        self.beta_theta = beta_theta
        self.v0 = nochoice_weight
        
    def getAttractionWeights(self, product_dict):
        '''
        Compute MNL attraction weights for all products.
        
        The attraction weight for product j is:
            w_j = exp(beta_0 + beta_p * p_j + sum_k beta_k * theta_{jk})
        
        Parameters
        ----------
        product_dict : dict
            Dictionary mapping product IDs to Product objects
            
        Returns
        -------
        mnl_weights : dict
            Dictionary mapping product IDs to attraction weights
        '''
        mnl_weights = {}
        for (j,p) in product_dict.items():
            # V_j = beta_0 + beta_price * price + sum_k beta_k * theta_k
            V_j = (self.beta_0 
                   + self.beta_price * p.theta[0] 
                   + np.sum([self.beta_theta[k] * p.theta[k+1] 
                            for k in range(self.num_attributes)]))
            mnl_weights[j] = np.exp(V_j)
        return mnl_weights

class ChoiceModel():
    '''
    Contextual Concavity (CC) Choice Model.
    
    The CC model captures reference-dependent choice behavior where the
    utility of product j depends on the reference point (minimum attribute
    values in the assortment):
    
        V_j = a_j - b_j * p_j + M_j
        
    where the contextual modifier M_j is:
    
        M_j = -gamma_0 * (p_j - p_min)^{eta_0} 
              + sum_{k=1}^{K} gamma_k * (theta_{jk} - theta_{min,k})^{eta_k}
    
    The gamma parameters control the strength of loss aversion (price) and
    gain seeking (non-price attributes). The eta parameters control the
    curvature (eta=1 is the Multi-attribute Loss Aversion special case).
    
    See Section 2 of the paper for full model specification.
    
    Attributes
    ----------
    num_levels : int
        Number of discrete levels for non-price attributes (L)
    max_level : dict
        Maximum level for each non-price attribute (for normalization)
    gamma : list
        Context sensitivity parameters [gamma_0, gamma_1, ..., gamma_K]
        - gamma_0: loss aversion for price
        - gamma_k: gain seeking for non-price attribute k
    eta : list
        Curvature parameters [eta_0, eta_1, ..., eta_K]
        - eta=1 corresponds to MLA (Multi-attribute Loss Aversion)
        - eta<1 corresponds to diminishing sensitivity
    a_dict : dict
        Product-specific intercepts {product_id: a_j}
    b_dict : dict
        Product-specific price sensitivities {product_id: b_j}
    v0 : float
        No-purchase utility weight
    num_attributes : int
        Total number of attributes including price (K+1)
    '''
    
    def __init__(self, num_levels, max_level, gamma, eta, a_dict, b_dict, nochoice_weight):
        '''
        Initialize a CC choice model.
        
        Parameters
        ----------
        num_levels : int
            Number of discrete levels for non-price attributes (L)
        max_level : dict
            Maximum level for each non-price attribute k: {k: max_level_k}
        gamma : list
            Context sensitivity parameters [gamma_0, gamma_1, ..., gamma_K]
        eta : list
            Curvature parameters [eta_0, eta_1, ..., eta_K]
        a_dict : dict
            Product-specific intercepts {product_id: a_j}
        b_dict : dict
            Product-specific price sensitivities {product_id: b_j}
        nochoice_weight : float
            No-purchase utility weight (v_0)
        '''
        self.num_levels = num_levels
        self.max_level = max_level
        self.gamma = gamma
        self.eta = eta
        self.v0 = nochoice_weight
        self.num_attributes = len(gamma) # K+1 (including price)
        self.a_dict = a_dict
        self.b_dict = b_dict
        
        
    def getNumAttributes(self):
        '''Return the number of attributes (K+1, including price).'''
        return self.num_attributes
    
    def getNumLevels(self):
        '''Return the number of discrete levels for non-price attributes (L).'''
        return self.num_levels
    
    def getAttractionWeights(self, reflist, product_dict):
        '''
        Compute CC attraction weights given a reference list.
        
        This function is used by the MILP solver. Given a fixed reference
        list (which determines the minimum attribute values), it computes
        the attraction weight for each product.
        
        Parameters
        ----------
        reflist : tuple
            Reference list where:
            - reflist[0] = product ID of price reference (lowest price product)
            - reflist[k] for k >= 1 = reference level for attribute k
        product_dict : dict
            Dictionary mapping product IDs to Product objects
            
        Returns
        -------
        attr_weights : dict
            Dictionary mapping product IDs to attraction weights
        '''
        # Build the reference point (minimum theta values)
        min_theta = [0]*len(reflist)
        min_theta[0] = product_dict[reflist[0]].theta[0] # Price of reference product
        for k in range(1,len(reflist)):
            min_theta[k] = reflist[k] # Reference level for non-price attribute k

        # Compute attraction weights for all products
        attr_weights = {}
        for (j,p) in product_dict.items():
            # Contextual modifier M_j (Equation 1 in paper)
            # Price component: -gamma_0 * (p_j - p_min)^{eta_0}
            price_loss = -self.gamma[0] * max(0, p.theta[0] - min_theta[0])**self.eta[0]

            # Non-price components: +gamma_k * (theta_{jk} - theta_{min,k})^{eta_k}
            attr_gains = np.sum([
                self.gamma[k] * max(0, (p.theta[k] - min_theta[k]) * self.max_level[k])**self.eta[k]
                for k in range(1, self.num_attributes)
            ])

            M = price_loss + attr_gains

            # Full utility: V_j = a_j - b_j * p_j + M_j
            attr_weights[j] = np.exp(self.a_dict[j] - self.b_dict[j] * p.theta[0] + M)
            
        return attr_weights
        
    def evaluateAssortment(self, assortment, product_dict, **kwargs):
        '''
        Evaluate an assortment under the CC model.
        
        Computes choice probabilities, expected profit, and consumer surplus
        for a given assortment. The reference point is determined by the
        minimum attribute values in the assortment.
        
        Parameters
        ----------
        assortment : list
            List of product IDs in the assortment
        product_dict : dict
            Dictionary mapping product IDs to Product objects
        **kwargs : optional
            - ref: list of reference products (if pre-specified)
            - price: dict of prices (if different from product.theta[0])
            
        Returns
        -------
        exp_profit : float
            Expected profit = sum_j margin_j * P_j(S)
        prob_dict : dict
            Choice probabilities {product_id: P_j(S)}
        consumer_surplus : float
            Expected consumer surplus = log(v_0 + sum_j V_j) + 0.57
            (0.57 is the Euler-Mascheroni constant)
        ref_products : list
            Reference products for each attribute [ref_0, ref_1, ..., ref_K]
        '''
        
        # Step 1: Determine the reference point (minimum attribute values)
        if kwargs and "ref" in kwargs.keys():
            # Reference products are pre-specified
            ref_products = kwargs["ref"]
            min_theta = [product_dict[ref_products[k]].theta[k]
                         for k in range(self.num_attributes)]
        else:
            # Find reference products (those with minimum attribute values)
            ref_products = [0]*(self.num_attributes)
            min_theta = [0]*self.num_attributes
            
            if kwargs and "price" in kwargs.keys():
                # Use provided prices instead of product.theta[0]
                price = kwargs["price"]

                # Find price reference (minimum price)
                min_theta[0] = price[assortment[0]]
                ref_products[0] = assortment[0]
                for j in assortment:
                    if price[j] < min_theta[0]:
                        min_theta[0] = price[j]
                        ref_products[0] = j

                # Find non-price attribute references (minimum levels)
                for k in range(1,self.num_attributes):
                    min_theta[k] = product_dict[assortment[0]].theta[k]
                    ref_products[k] = assortment[0]
                    for j in assortment:
                        if product_dict[j].theta[k] < min_theta[k]:
                            min_theta[k] = product_dict[j].theta[k]
                            ref_products[k] = j
            else:
                # Use product.theta[0] as prices
                for k in range(self.num_attributes):
                    min_theta[k] = product_dict[assortment[0]].theta[k]
                    ref_products[k] = assortment[0]
                    for j in assortment:
                        if product_dict[j].theta[k] < min_theta[k]:
                            min_theta[k] = product_dict[j].theta[k]
                            ref_products[k] = j
                        
        # Step 2: Compute attraction weights for products in assortment
        attr_weights = []
        margins = []
        for j in assortment:
            p = product_dict[j]

            # Get price (either from kwargs or from product attributes)
            if kwargs and "price" in kwargs.keys():
                price_j = kwargs["price"][j]
            else:
                price_j = p.theta[0]

            # Contextual modifier M_j
            price_loss = -self.gamma[0] * max(0, price_j - min_theta[0])**self.eta[0]
            attr_gains = np.sum([
                self.gamma[k] * max(0, (p.theta[k] - min_theta[k]) * self.max_level[k])**self.eta[k]
                for k in range(1, self.num_attributes)
            ])
            M_j = price_loss + attr_gains
            
            # Attraction weight
            w = np.exp(self.a_dict[j] - self.b_dict[j] * price_j + M_j)
            attr_weights.append(w)
            margins.append(p.margin)

        # Step 3: Compute CC choice probabilities
        tot_weight = np.sum(attr_weights)
        attr_weights = np.array(attr_weights)
        cc_probs = attr_weights/(self.v0 + tot_weight)

        # Step 4: Compute expected profit and consumer surplus
        prob_dict = {assortment[i]: cc_probs[i] for i in range(len(cc_probs))}
        exp_profit = np.inner(cc_probs, margins)

        # Consumer surplus: E[max utility] = log(sum of weights) + Euler constant
        consumer_surplus = np.log(self.v0 + np.sum(attr_weights)) + 0.57
        
        return (exp_profit, prob_dict, consumer_surplus, ref_products)    
                



