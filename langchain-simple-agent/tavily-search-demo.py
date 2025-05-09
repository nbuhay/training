from langchain_community.tools.tavily_search import TavilySearchResults

import os
os.environ["TAVILY_API_KEY"] = '<API-KEY>'  # Replace with your actual API key

search = TavilySearchResults(max_results=2)
search_results = search.invoke("latest Verizon news")
print(search_results)
# If we want, we can create other tools.
# Once we have all the tools we want, we can put them in a list that we will reference later.
tools = [search]

# result
# [{'title': 'News Releases About Verizon', 'url': 'https://www.verizon.com/about/news/news-releases', 'content': "Verizon to report 1Q earnings on April 22, 2025\n\nVerizon will report Q1 2025 earnings on April 22 at 8:30 a.m. ET via webcast. Materials available at 7:00 a.m. on Verizon’s Investor Relations site: verizon.com/about/investors.\n\nVerizon announces Rescue 42 as latest “Verizon Frontline Verified” partner [...] Verizon's 2025 DBIR reveals a 100% surge in EMEA data breaches, with system intrusions at 53% and insider leaks at 29%. Emphasizes need for stronger internal cybersecurity and employee training.\n\nVerizon delivered strong financial growth with industry-leading wireless service revenue in 1Q 2025\n\nCustomer segmentation strategy is a key driver of successful financial performance. Verizon remains confident in full-year 2025 guidance. Fueled by innovative and segmented product offerings. [...] Verizon announced the availability of the Verizon Frontline Network Slice in select markets nationwide, continuing to build on the company’s more than 30-year history of cutting-edge innovation.\n\nVerizon’s 2025 Data Breach Investigations Report: Alarming surge in cyberattacks through third-parties\n\nVerizon’s 2025 Data Breach Investigations Report reveals a significant increase in cyberattacks & third-party breaches. Learn key findings and insights.", 'score': 0.7447496}, {'title': 'Verizon Sourcing LLC - Press Release Distribution and Management', 'url': 'https://www.globenewswire.com/search/organization/Verizon%2520Sourcing%2520LLC', 'content': "NEW YORK, April 24, 2025 (GLOBE NEWSWIRE) -- [The big news] The new motorola razr is coming to Verizon, and it's smarter, sleeker and more iconic than ever.", 'score': 0.7036111}]