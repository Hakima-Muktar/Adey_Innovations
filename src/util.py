import pandas as pd
import numpy as np
import os

def ip_to_int(ip):
    """Convert an IP address to integer format."""
    try:
        parts = list(map(int, str(ip).split('.')))
        if len(parts) != 4:
            return np.nan
        return (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]
    except Exception:
        return np.nan

def map_ip_to_country(df, country_df):
    """
    Efficiently map IP addresses to countries.
    """
    df_copy = df.copy()
    df_copy['ip_int'] = df_copy['ip_address'].apply(ip_to_int)
    
    # Drop rows with invalid IPs for the merge
    valid_ips = df_copy.dropna(subset=['ip_int'])
    invalid_ips = df_copy[df_copy['ip_int'].isna()]
    
    # Sort country_df for merge_asof
    country_df = country_df.sort_values('lower_bound_ip_address')
    
    # Perform merge_asof
    merged_df = pd.merge_asof(
        valid_ips.sort_values('ip_int'),
        country_df,
        left_on='ip_int',
        right_on='lower_bound_ip_address',
        direction='backward'
    )
    
    # Filter where ip_int is within the range
    mask = (merged_df['ip_int'] >= merged_df['lower_bound_ip_address']) & \
           (merged_df['ip_int'] <= merged_df['upper_bound_ip_address'])
    
    merged_df.loc[~mask, 'country'] = 'Unknown'
    
    # Add back invalid IPs as Unknown
    if len(invalid_ips) > 0:
        invalid_ips['country'] = 'Unknown'
        merged_df = pd.concat([merged_df, invalid_ips], ignore_index=True)
    
    return merged_df

def save_stats(stats_df, filename):
    """Save summary statistics to the stats directory."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    path = os.path.join(base_dir, 'report/stats', filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stats_df.to_csv(path, index=True)

def save_plot(plt_obj, filename):
    """Save plot to the images directory."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    path = os.path.join(base_dir, 'report/images', filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt_obj.savefig(path, bbox_inches='tight')